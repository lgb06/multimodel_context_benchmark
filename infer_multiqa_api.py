"""Inference script for DocGenome converted multiqa dataset.

Input format (json/jsonl):
    {"images":[...], "query":"...<image>...", "response":"..."}

Output format (jsonl):
    {
      "idx": int,
      "query": str,
      "images": [...],
      "ground_truth": str,
      "model_output": str,
      "model": str,
      "error": str | null
    }

Usage examples:
    python infer_multiqa_api.py --model gpt-5.1 \
      --input qa_info/docgenome_testset_multiqa_converted.jsonl

    python infer_multiqa_api.py --model deepseek-chat \
      --base-url https://api.deepseek.com/v1 \
      --api-key YOUR_KEY \
      --workers 8
"""

import argparse
import base64
import json
import mimetypes
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm


def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"[{get_timestamp()}] {message}")


def load_input(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if path.suffix.lower() == ".jsonl":
        data = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError("JSON input must be a list.")
    return obj


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    path = Path(file_path)
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def append_jsonl(item: Dict[str, Any], file_path: str) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def encode_image(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception:
        return None


def to_data_url(image_path: str) -> Optional[str]:
    encoded = encode_image(image_path)
    if not encoded:
        return None
    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        mime = "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def build_multimodal_messages(query: str, images: List[str], system_prompt: str) -> List[Dict[str, Any]]:
    content = []
    parts = query.split("<image>")

    for i, part in enumerate(parts):
        if part.strip():
            content.append({"type": "text", "text": part.strip()})
        if i < len(images):
            url = to_data_url(images[i])
            if url:
                content.append({"type": "image_url", "image_url": {"url": url}})
            else:
                content.append({"type": "text", "text": f"[Image load failed: {images[i]}]"})

    if len(images) > len(parts) - 1:
        for extra in images[len(parts) - 1 :]:
            url = to_data_url(extra)
            if url:
                content.append({"type": "image_url", "image_url": {"url": url}})

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def call_openai_api(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    model: str,
    max_retries: int = 3,
    retry_delay: int = 3,
) -> Tuple[Optional[str], Optional[str]]:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return response.choices[0].message.content, None
        except Exception as e:
            err = str(e)
            if attempt < max_retries - 1:
                log(f"   retry {attempt + 1}/{max_retries - 1}: {err[:160]}")
                time.sleep(retry_delay)
            else:
                return None, err
    return None, "unknown error"


def process_single_case(args: tuple) -> Tuple[int, Dict[str, Any]]:
    idx, item, client, model, retry_delay, max_retries, system_prompt = args
    query = item.get("query", "")
    images = item.get("images", [])
    ground_truth = item.get("response", "")

    if not query:
        return idx, {
            "idx": idx,
            "query": query,
            "images": images,
            "ground_truth": ground_truth,
            "model_output": None,
            "model": model,
            "error": "missing query",
        }

    messages = build_multimodal_messages(query, images, system_prompt)
    output, error = call_openai_api(
        client,
        messages,
        model=model,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )

    return idx, {
        "idx": idx,
        "query": query,
        "images": images,
        "ground_truth": ground_truth,
        "model_output": output,
        "model": model,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DocGenome multiqa inference with OpenAI-compatible API.")
    parser.add_argument("--model", type=str, required=True, default="gemini-3.1-pro-preview-thinking", help="Model name.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/input.json",
        help="Input dataset path (.json or .jsonl).",
    )
    parser.add_argument("--output", type=str, default="outputs/output.jsonl", help="Output jsonl path.")
    parser.add_argument("--base-url", type=str, default='http://35.220.164.252:3888/v1/', help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", type=str, default='sk-UJ9PZW2UAT7ja3aM0qFwinXdAPXkrYojnhfG7D28lYOdQrwu', help="API key (or use OPENAI_API_KEY).")
    parser.add_argument("--workers", type=int, default=100, help="Concurrent workers.")
    parser.add_argument("--max-samples", type=int, default=None, help="Only run first N samples.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per sample.")
    parser.add_argument("--retry-delay", type=int, default=3, help="Retry delay seconds.")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful assistant that can analyze images and text to answer questions.",
        help="System prompt.",
    )
    args = parser.parse_args()

    if args.output is None:
        model_name_safe = args.model.replace("/", "_").replace(":", "_")
        args.output = f"outputs/{model_name_safe}.jsonl"

    log(f"input: {args.input}")
    log(f"output: {args.output}")
    log(f"model: {args.model}")
    log(f"workers: {args.workers}")

    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Use --api-key or set OPENAI_API_KEY.")

    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
        log(f"base_url: {args.base_url}")
    client = OpenAI(**client_kwargs)

    data = load_input(args.input)
    if args.max_samples is not None:
        data = data[: args.max_samples]
    log(f"samples: {len(data)}")

    completed = set()
    existing = load_jsonl(args.output)
    for item in existing:
        idx = item.get("idx")
        if isinstance(idx, int):
            completed.add(idx)
    if completed:
        log(f"resume: {len(completed)} already done")

    tasks = []
    for idx, item in enumerate(data):
        if idx in completed:
            continue
        tasks.append((idx, item, client, args.model, args.retry_delay, args.max_retries, args.system_prompt))

    if not tasks:
        log("all samples completed")
        return

    success = 0
    failed = 0

    if args.workers == 1:
        for task in tqdm(tasks, desc="inference"):
            _, result = process_single_case(task)
            append_jsonl(result, args.output)
            if result["error"] is None:
                success += 1
            else:
                failed += 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_single_case, t) for t in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="inference"):
                _, result = future.result()
                append_jsonl(result, args.output)
                if result["error"] is None:
                    success += 1
                else:
                    failed += 1

    log("=" * 40)
    log(f"done. success={success}, failed={failed}")
    log(f"saved: {args.output}")


if __name__ == "__main__":
    main()
