#!/usr/bin/env python3
"""FTEC5660 HW1 student starter: build a chain for supermarket receipts."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


QUERY_1 = "How much money did I spend in total for these bills?"
QUERY_2 = "How much would I have had to pay without the discount?"
QUERIES = (QUERY_1, QUERY_2)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
DUMMY_RESPONSE = "please design your chain to answer these two queries."


def load_env_file(path: Path = Path(".env")) -> None:
    """Load the simple KEY=VALUE entries used by this homework."""
    if not path.is_file():
        return
    import os

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def image_files(folder: Path) -> list[Path]:
    """Return supported images directly inside *folder*, sorted by filename."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_data_url(path: Path) -> str:
    """Encode a local image in the format accepted by a multimodal prompt."""
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_chain() -> Any:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_deepseek import ChatDeepSeek
    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        max_retries=2,
    )
    task = """
Read one supermarket receipt and return JSON only:
{{"final_payment":"102.30","subtotal":"102.31",
"discounts":["5.39"],"check_without_discount":"107.70"}}
final_payment: actual payment after ROUNDING.
subtotal: SUBTOTAL before ROUNDING.
discounts: positive amounts of all discounts, promotions, coupons and savings.
Exclude ROUNDING, change, points, balances and item prices.
check_without_discount = subtotal + sum(discounts).
Use decimal strings only.
""".strip()
    extract_prompt = ChatPromptTemplate.from_messages([
        ("system", "Accurately extract amounts from English or Chinese receipts."),
        ("human", [
            {"type": "text", "text": task},
            {
                "type": "image_url",
                "image_url": {"url": "{image_url}"},
            },
        ]),
    ])
    repair_prompt = ChatPromptTemplate.from_messages([
        ("system", "Check the receipt and correct the draft JSON."),
        ("human", [
            {
                "type": "text",
                "text": task + "\nDraft:\n{draft}",
            },
            {
                "type": "image_url",
                "image_url": {"url": "{image_url}"},
            },
        ]),
    ])
    parser = StrOutputParser()
    return {
        "extract": (
            extract_prompt | model | parser
        ).with_retry(stop_after_attempt=2),
        "repair": (
            repair_prompt | model | parser
        ).with_retry(stop_after_attempt=2),
    }


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    def parse(raw: Any) -> tuple[Decimal, Decimal]:
        text = response_text(raw)
        data = json.loads(
            text[text.find("{"):text.rfind("}") + 1]
        )
        def money(value: Any) -> Decimal:
            match = re.search(
                r"-?\d[\d,]*(?:\.\d+)?",
                str(value),
            )
            if not match:
                raise ValueError
            return Decimal(
                match.group().replace(",", "")
            )
        paid = money(data["final_payment"])
        subtotal = money(data["subtotal"])
        discounts = sum(
            (
                abs(money(value))
                for value in data["discounts"]
            ),
            Decimal("0"),
        )
        full_price = subtotal + discounts
        if abs(
            full_price
            - money(data["check_without_discount"])
        ) > Decimal("0.01"):
            raise ValueError
        return paid, full_price
    inputs = [
        {"image_url": image_data_url(image)}
        for image in images
    ]
    outputs = chain["extract"].batch(
        inputs,
        config={"max_concurrency": 3},
        return_exceptions=True,
    )
    total_paid = Decimal("0")
    total_full_price = Decimal("0")
    for values, output in zip(inputs, outputs):
        try:
            paid, full_price = parse(output)
        except Exception:
            try:
                repaired = chain["repair"].invoke({
                    "image_url": values["image_url"],
                    "draft": response_text(output),
                })
                paid, full_price = parse(repaired)
            except Exception:
                paid = Decimal("0")
                full_price = Decimal("0")
        total_paid += paid
        total_full_price += full_price
    return {
        QUERY_1: f"HK${total_paid:.2f}",
        QUERY_2: f"HK${total_full_price:.2f}",
    }


# Everything below is provided runner/scoring code. No edits are needed.

_MONEY_RE = re.compile(
    r"(?<![\w.])(?:HK\$|\$)?\s*(-?\d[\d,]*(?:\.\d+)?)(?![\w.])",
    re.IGNORECASE,
)


def response_text(value: Any) -> str:
    """Convert common LangChain response shapes to text for results.csv."""
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts).strip()
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=False)
    return str(content).strip()


def parse_single_amount(text: str) -> Decimal | None:
    """Accept a response only when it contains exactly one numeric amount."""
    matches = _MONEY_RE.findall(text)
    if len(matches) != 1:
        return None
    try:
        return Decimal(matches[0].replace(",", "")).quantize(Decimal("0.01"))
    except InvalidOperation:
        return None


def read_ground_truth(folder: Path) -> dict[str, Decimal]:
    """Read aggregate answers from the test folder."""
    path = folder / "ground_truth.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    answers = data.get("answers", data)
    return {query: Decimal(str(answers[query])).quantize(Decimal("0.01")) for query in QUERIES}


def correctness_text(response: str, expected: Decimal | None) -> str:
    """Return `correct`, or an expected/predicted mismatch explanation."""
    if expected is None:
        return "not graded: ground_truth.json is missing"
    predicted = parse_single_amount(response)
    if predicted == expected:
        return "correct"
    shown = f"HK${predicted:.2f}" if predicted is not None else repr(response)
    return f"incorrect: expected HK${expected:.2f}, predicted {shown}"


def write_results(responses: dict[str, Any], truth: dict[str, Decimal]) -> Path:
    """Write the required three-column results.csv file."""
    output = Path("results.csv")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["query", "model_response", "correctness"])
        for query in QUERIES:
            text = response_text(responses.get(query, "<missing response>"))
            writer.writerow([query, text, correctness_text(text, truth.get(query))])
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FTEC5660 HW1 on receipt images")
    parser.add_argument(
        "--image-folder",
        required=True,
        type=Path,
        help="folder containing supermarket receipt images",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image_folder.is_dir():
        raise SystemExit(f"not a folder: {args.image_folder}")

    images = image_files(args.image_folder)
    if not images:
        raise SystemExit(f"no supported images found in {args.image_folder}")

    load_env_file()
    chain = build_chain()
    responses = answer_queries(chain, images)
    if not isinstance(responses, dict):
        raise TypeError("answer_queries() must return a dictionary")

    output = write_results(responses, read_ground_truth(args.image_folder))
    print(f"Processed {len(images)} receipt(s). Wrote {output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
