import asyncio
import base64
import io
import json
import logging
import pickle

# import random
import time

# from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

import boto3

# import botocore
import pandas as pd
from anthropic import AsyncAnthropicBedrock, RateLimitError
from anthropic.types.message import Message

# from anthropic.types.tool_use_block import ToolUseBlock
from dotenv import load_dotenv
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFPageCountError
from PIL import Image
from PIL.Image import DecompressionBombError
from tqdm.asyncio import tqdm

# from tqdm.notebook import tqdm as tqdm_notebook

base_prompt = """
Is this a form? Answer Yes or No. 
It's only a form if it contains form field boxes.
Hand drawn forms, questionnaires and surveys are all valid forms.
If it is a form, extract the questions from it using the extract_form_questions tool.
If there is no output, explain why.
"""


async def test_client(client):
    response = await client.messages.create(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        temperature=0.01,
        max_tokens=2,
        messages=[{"role": "user", "content": "reply Yes. Nothing else"}],
    )

    return response.content[0].text == "Yes."


def get_all_files(folder_path: str) -> List[Path]:
    """Recursive file list

    Args:
        folder_path (str): location to be crawled

    Returns:
        List[Path]: A list of paths for all files in `folder_path`
    """
    return list(Path(folder_path).glob("**/*"))


def load_batch_results(results_file: str) -> Dict[str, Dict]:
    try:
        with open(results_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def save_batch_results(results: Dict[str, Dict], results_file: str):
    with open(results_file, "wb") as f:
        pickle.dump(results, f)


def get_all_files(folder_path: str) -> List[Path]:
    return list(Path(folder_path).glob("**/*"))


def pdf_to_image_bytes(pdf_path: Path, width: int = 600, dpi: int = 300):
    # Convert PDF to list of PIL Image objects
    logging.info(f"Converting {pdf_path} to images")
    try:
        images = convert_from_path(pdf_path, dpi=dpi)
    except PDFPageCountError as e:
        logging.warning(f"{pdf_path} encountered an error return empty list {e}")
        return []
    except DecompressionBombError as e:
        logging.warn(
            f"{pdf_path} encountered a decompression bomb error, returning empty list: {e}"
        )
        return []

    image_bytes_list = []

    for i, img in enumerate(images):
        # Resize image if width is specified
        if width:
            ratio = width / float(img.width)
            height = int(ratio * img.height)
            img = img.resize((width, height), Image.LANCZOS)

        # Convert PIL Image to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        img_byte_arr = img_byte_arr.getvalue()

        image_bytes_list.append(img_byte_arr)

    return image_bytes_list


async def exponential_backoff(attempt, base_delay):
    delay = base_delay * (2**attempt)
    await asyncio.sleep(delay)
    return delay


def encode_image(byte_array):
    """encode image for claude"""
    return base64.b64encode(byte_array).decode("utf-8")


def format_messages(image_bytes):
    messages = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": encode_image(img),
            },
        }
        for img in image_bytes
    ] + [{"type": "text", "text": base_prompt}]
    return messages


async def ask_claude(messages, client, extraction_tool):
    response = await client.messages.create(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        temperature=0.01,
        max_tokens=5000,
        tools=[extraction_tool],
        messages=[{"role": "user", "content": messages}],
    )
    return {
        "result": response,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }


async def get_message_with_backoff(
    messages, semaphore, client, extraction_tool, max_retries=5, base_delay=1
):
    async with semaphore:
        for attempt in range(max_retries):
            try:
                return await ask_claude(messages, client, extraction_tool)
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    logging.exception(
                        f"Rate limit exceeded after {max_retries} attempts | Error: {e}"
                    )
                    return {
                        "result": e.status_code,
                        "input_tokens": None,
                        "output_tokens": None,
                    }
                delay = await exponential_backoff(attempt, base_delay)
                logging.warning(f"Rate limit hit, retrying in {delay:.2f} seconds...")
            except Exception as e:
                logging.exception(f"Exception occurred | Error: {e}")
                return {"result": str(e), "input_tokens": None, "output_tokens": None}


async def process_form(
    pdf_path: Path, semaphore: asyncio.Semaphore, client, extraction_tool, pbar: tqdm
) -> Dict:
    task_id = id(asyncio.current_task())

    async with semaphore:
        start_time = time.time()
        active_tasks = len([task for task in asyncio.all_tasks() if not task.done()])
        max_tasks = semaphore._value
        logging.info(
            f"start_processing:{pdf_path} (task_id: {task_id}, active_tasks: {active_tasks}/{max_tasks})"
        )

        # Move PDF conversion inside the semaphore
        images = pdf_to_image_bytes(pdf_path, 600, 300)
        logging.info(
            f"[{time.time():.3f}] Task {task_id} - {pdf_path}: Converted to images."
        )

        if len(images) > 19:
            images = images[0:19]
            logging.info(
                f"[{time.time():.3f}] Task {task_id} - {pdf_path}: truncated as too long"
            )

        messages = format_messages(images)

        logging.info(
            f"sending_request:{pdf_path} (task_id: {task_id}, active_tasks: {active_tasks}/{max_tasks})"
        )
        result = await get_message_with_backoff(
            messages, semaphore, client, extraction_tool
        )

        end_time = time.time()
        processing_time = end_time - start_time
        logging.info(
            f"processing_complete:{pdf_path} (task_id: {task_id}, time: {processing_time:.2f}s)"
        )

        # Update the progress bar
        pbar.update(1)

        return result


async def process_batch(
    batch: List[Path], semaphore: asyncio.Semaphore, client, extraction_tool
) -> Dict[str, Dict]:
    results = {}
    for pdf_path in batch:
        try:
            result = await process_form(pdf_path, semaphore, client, extraction_tool)
            results[str(pdf_path)] = result
        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {str(e)}")
    return results


async def process_forms_in_batches(
    folder_path: str,
    client,
    extraction_tool,
    batch_size: int = 10,
    max_concurrent: int = 5,
    results_file: str = "batch_results.pickle",
) -> Dict[str, Dict]:
    all_files = get_all_files(folder_path)
    pdf_files = [Path(file) for file in all_files if str(file).lower().endswith(".pdf")]

    semaphore = asyncio.Semaphore(max_concurrent)

    # Load existing results if any
    results = load_batch_results(results_file)

    # Filter out already processed files
    pdf_files = [file for file in pdf_files if str(file) not in results]

    # Create progress bar
    pbar = tqdm(total=len(pdf_files), desc="Processing Forms", unit="form")

    async def process_form_wrapper(pdf_path):
        result = await process_form(pdf_path, semaphore, client, extraction_tool, pbar)
        return str(pdf_path), result

    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i : i + batch_size]

        logging.info(
            f"[{time.time():.3f}] Starting batch {i//batch_size + 1} of {len(pdf_files)//batch_size + 1}"
        )

        # Process the batch asynchronously, but limit concurrency
        batch_start_time = time.time()
        tasks = [process_form_wrapper(pdf_path) for pdf_path in batch]
        batch_results = dict(await asyncio.gather(*tasks))

        batch_end_time = time.time()

        # Update results after the batch is complete
        results.update(batch_results)

        # Save results after the entire batch is processed
        save_batch_results(results, results_file)

        batch_processing_time = batch_end_time - batch_start_time
        logging.info(
            f"[{time.time():.3f}] Completed batch {i//batch_size + 1} in {batch_processing_time:.2f} seconds"
        )

        logging.info(
            f"[{time.time():.3f}] Processed and saved batch {i//batch_size + 1} of {len(pdf_files)//batch_size + 1}"
        )

    pbar.close()
    return results


def run_form_processing(
    folder_path: str,
    client,
    extraction_tool,
    batch_size: int = 10,
    max_concurrent: int = 5,
    results_file: str = "batch_results.pickle",
) -> pd.DataFrame:
    print("Processing forms. Check 'form_processing.log' for detailed logs.")

    async def run_async():
        return await process_forms_in_batches(
            folder_path,
            client,
            extraction_tool,
            batch_size,
            max_concurrent,
            results_file,
        )

    results = asyncio.run(run_async())
    print(f"Processed {len(results)} forms")
    return results


def compute_total(results_df):
    total_count = results_df[["input_tokens", "output_tokens"]].sum()
    print(total_count)
    rates_per1000 = {"input_cost": 0.003, "output_cost": 0.015}

    return (
        total_count["input_tokens"] * rates_per1000["input_cost"] / 1000
        + total_count["output_tokens"] * rates_per1000["output_cost"] / 1000
    )
