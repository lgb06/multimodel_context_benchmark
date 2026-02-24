import argparse
import base64
import datetime as dt
import http.client
import json
import mimetypes
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send text + image(s) from the last record in input JSON."
    )
    parser.add_argument(
        "--input-json",
        default="data/input.json",
        help="Input JSON file. The last object is used.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Model id, for example gemini-2.5-flash or your gemini-3 model id.",
    )
    parser.add_argument(
        "--scheme",
        choices=("http", "https"),
        default=os.getenv("GEMINI_API_SCHEME", "http"),
        help="API scheme.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("GEMINI_API_HOST", "35.220.164.252"),
        help="API host.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("GEMINI_API_PORT", "3888")),
        help="API port.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("GEMINI_API_TOKEN", ""),
        help="Bearer token. Prefer env var GEMINI_API_TOKEN.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Directory to save raw response and run logs.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("GEMINI_API_TIMEOUT", "120")),
        help="HTTP timeout seconds.",
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


def load_image_as_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def build_payload(text: str, image_paths: list[Path]) -> dict:
    parts = [{"text": text}]
    for path in image_paths:
        parts.append(
            {
                "inline_data": {
                    "mime_type": get_mime_type(path),
                    "data": load_image_as_base64(path),
                }
            }
        )
    return {"contents": [{"parts": parts}]}


def extract_model_text(response_json: dict) -> str:
    output_parts = []
    for candidate in response_json.get("candidates", []):
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            text = part.get("text")
            if text:
                output_parts.append(text)
    return "\n".join(output_parts).strip()


def save_outputs(
    out_dir: Path,
    status_code: int,
    request_path: str,
    model: str,
    prompt_text: str,
    image_paths: list[Path],
    response_json: dict | None,
    raw_response_text: str,
    model_output_text: str,
    metadata: dict,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_response_file = out_dir / f"gemini_response_{timestamp}.json"
    log_file = out_dir / "gemini_run_log.jsonl"

    response_to_save = response_json if response_json is not None else {"raw": raw_response_text}
    with raw_response_file.open("w", encoding="utf-8") as f:
        json.dump(response_to_save, f, ensure_ascii=False, indent=2)

    log_entry = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "status_code": status_code,
        "request_path": request_path,
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
    if not args.token:
        raise SystemExit(
            "Missing token. Set GEMINI_API_TOKEN or pass --token."
        )

    prompt_text, image_paths, metadata = load_latest_input_record(Path(args.input_json))
    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not image_path.is_file():
            raise ValueError(f"Image path is not a file: {image_path}")

    payload = build_payload(prompt_text, image_paths)
    request_path = f"/v1beta/models/{args.model}:generateContent"
    payload_text = json.dumps(payload)

    connection_cls = (
        http.client.HTTPSConnection if args.scheme == "https" else http.client.HTTPConnection
    )
    conn = connection_cls(args.host, args.port, timeout=args.timeout)
    headers = {
        "Authorization": f"Bearer {args.token}",
        "Content-Type": "application/json",
    }

    try:
        conn.request("POST", request_path, payload_text, headers)
        response = conn.getresponse()
        raw_response_text = response.read().decode("utf-8", errors="replace")
        status_code = response.status
    finally:
        conn.close()

    try:
        response_json = json.loads(raw_response_text)
    except json.JSONDecodeError:
        response_json = None

    model_output_text = (
        extract_model_text(response_json) if isinstance(response_json, dict) else raw_response_text
    )
    raw_file, log_file = save_outputs(
        out_dir=Path(args.out_dir),
        status_code=status_code,
        request_path=request_path,
        model=args.model,
        prompt_text=prompt_text,
        image_paths=image_paths,
        response_json=response_json if isinstance(response_json, dict) else None,
        raw_response_text=raw_response_text,
        model_output_text=model_output_text,
        metadata=metadata,
    )

    print(f"HTTP status: {status_code}")
    print(f"Input JSON: {metadata['input_json']}")
    print(f"Source: {metadata['source']}")
    print(f"Difficulty: {metadata['difficulty']}")
    print(f"Annotation: {metadata['annotation']}")
    print(f"Raw response file: {raw_file}")
    print(f"Run log file: {log_file}")
    print("\nModel output:")
    print(model_output_text if model_output_text else "(empty)")

    return 0 if status_code < 400 else 1


if __name__ == "__main__":
    raise SystemExit(main())
