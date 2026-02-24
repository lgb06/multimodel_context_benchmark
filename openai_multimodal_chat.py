import argparse
import base64
import datetime as dt
import json
import mimetypes
import os
from pathlib import Path

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use OpenAI SDK to send text + image(s) from the last record in input JSON."
    )
    parser.add_argument(
        "--input-json",
        default="data/input.json",
        help="Input JSON file. The last object is used.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Model id, for example your gemini-3 model id.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", "http://35.220.164.252:3888/v1"),
        help="OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", os.getenv("GEMINI_API_TOKEN", "")),
        help="API key for OpenAI SDK. Prefer env var OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Directory to save raw response and run logs.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("OPENAI_TIMEOUT", "120")),
        help="Request timeout seconds.",
    )
    return parser.parse_args()


def load_latest_input_record(input_json_path: Path) -> tuple[str, list[Path], dict]:
    if not input_json_path.exists():
        raise FileNotFoundError(f"Input JSON not found: {input_json_path}")
    if not input_json_path.is_file():
        raise ValueError(f"Input JSON path is not a file: {input_json_path}")

    with input_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("Input JSON must be a non-empty array.")

    latest = data[-1]
    if not isinstance(latest, dict):
        raise ValueError("The last item in input JSON must be an object.")

    required_fields = ("text", "image", "source", "difficulty", "annotation")
    missing_fields = [field for field in required_fields if field not in latest]
    if missing_fields:
        raise ValueError(f"Missing fields in last input object: {', '.join(missing_fields)}")

    text = latest["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Field 'text' must be a non-empty string.")

    image_value = latest["image"]
    if isinstance(image_value, str):
        image_list = [image_value]
    elif isinstance(image_value, list) and image_value and all(
        isinstance(item, str) and item.strip() for item in image_value
    ):
        image_list = image_value
    else:
        raise ValueError("Field 'image' must be a non-empty string or non-empty string list.")

    metadata = {
        "source": latest["source"],
        "difficulty": latest["difficulty"],
        "annotation": latest["annotation"],
        "input_json": str(input_json_path),
    }
    return text, [Path(item) for item in image_list], metadata


def get_mime_type(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image file type: {image_path}")
    return mime_type


def load_image_as_data_url(image_path: Path) -> str:
    mime_type = get_mime_type(image_path)
    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_messages(text: str, image_paths: list[Path]) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": text}]
    for path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": load_image_as_data_url(path)},
            }
        )
    return [{"role": "user", "content": content}]


def extract_model_text(response_data: dict) -> str:
    output_parts = []
    for choice in response_data.get("choices", []):
        message = choice.get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            if content:
                output_parts.append(content)
            continue
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if text:
                        output_parts.append(text)
    return "\n".join(output_parts).strip()


def save_outputs(
    out_dir: Path,
    model: str,
    prompt_text: str,
    image_paths: list[Path],
    response_data: dict,
    model_output_text: str,
    metadata: dict,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_response_file = out_dir / f"openai_response_{timestamp}.json"
    log_file = out_dir / "openai_run_log.jsonl"

    with raw_response_file.open("w", encoding="utf-8") as f:
        json.dump(response_data, f, ensure_ascii=False, indent=2)

    log_entry = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "base_url": response_data.get("_base_url"),
        "model": model,
        "prompt_text": prompt_text,
        "image_paths": [str(p) for p in image_paths],
        "raw_response_file": str(raw_response_file),
        "model_output_text": model_output_text,
        "source": metadata["source"],
        "difficulty": metadata["difficulty"],
        "annotation": metadata["annotation"],
        "input_json": metadata["input_json"],
    }
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return raw_response_file, log_file


def main() -> int:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api-key.")

    prompt_text, image_paths, metadata = load_latest_input_record(Path(args.input_json))
    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not image_path.is_file():
            raise ValueError(f"Image path is not a file: {image_path}")

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )
    messages = build_messages(prompt_text, image_paths)

    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
    )
    response_data = response.model_dump()
    response_data["_base_url"] = args.base_url

    model_output_text = extract_model_text(response_data)
    raw_file, log_file = save_outputs(
        out_dir=Path(args.out_dir),
        model=args.model,
        prompt_text=prompt_text,
        image_paths=image_paths,
        response_data=response_data,
        model_output_text=model_output_text,
        metadata=metadata,
    )

    print("Status: success")
    print(f"Input JSON: {metadata['input_json']}")
    print(f"Source: {metadata['source']}")
    print(f"Difficulty: {metadata['difficulty']}")
    print(f"Annotation: {metadata['annotation']}")
    print(f"Raw response file: {raw_file}")
    print(f"Run log file: {log_file}")
    print("\nModel output:")
    print(model_output_text if model_output_text else "(empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
