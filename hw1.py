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
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_deepseek import ChatDeepSeek

    model = ChatDeepSeek(
        model="deepseek-v4-flash-vision-exp",
        temperature=0,
        max_tokens=1500,
        timeout=60,
        max_retries=1,
    )

    task = """
Read this receipt and return JSON only:
{{"paid":null,"subtotal":null,"rounding":null,"items":[],"discounts":[]}}
Use decimal strings for amounts, and arrays of decimal strings for the lists.
paid: final amount paid, not cash tendered, change, balance or duplicate payment.
subtotal: printed SUBTOTAL / 小計 before rounding.
rounding: signed ROUNDING adjustment; use "0.00" only when absent.
items: all posted item and fee totals, including plastic bags; preserve signs.
discounts: positive magnitudes of ALL applied discount lines, including coupons,
member/app offers, MB PRICE, packaging reductions and percentage discounts.
Copy actual rightmost posted amounts, not amounts inside promotional wording.
Keep repeated lines. Line totals already include quantity; do not multiply again.
Exclude subtotal, rounding, payments, change, balances and savings summaries
from both lists. Do not recalculate percentages or invent missing amounts.
Use null for unreadable values. Output no labels or explanations.
{feedback}
""".strip()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Read English and Chinese receipts accurately."),
        ("human", [
            {"type": "text", "text": task},
            {"type": "image_url", "image_url": {"url": "{image_url}"}},
        ]),
    ])
    return prompt | model | StrOutputParser()


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    zero = Decimal("0.00")

    def money(value):
        number = Decimal(str(value).replace(",", ""))
        if not number.is_finite() or number != number.quantize(Decimal("0.01")):
            raise ValueError("Invalid money value")
        return number

    def parse(text):
        data = json.loads(text[text.index("{"):text.rindex("}") + 1])
        subtotal = money(data["subtotal"])
        expected_paid = subtotal + money(data["rounding"])
        paid = money(data["paid"]) if data.get("paid") is not None else expected_paid
        if not all(isinstance(data[key], list) for key in ("items", "discounts")):
            raise ValueError("Expected amount lists")
        original = subtotal + sum((abs(money(x)) for x in data["discounts"]), zero)
        item_total = sum((money(x) for x in data["items"]), zero)
        return [(paid, paid == expected_paid), (original, original == item_total)]

    inputs = [{"image_url": image_data_url(path), "feedback": ""} for path in images]
    records = [[None, None] for _ in images]

    for _ in range(2):
        pending = [i for i, row in enumerate(records) if any(x is None or not x[1] for x in row)]
        if not pending:
            break
        try:
            outputs = chain.batch(
                [inputs[i] for i in pending],
                config={"max_concurrency": 3},
                return_exceptions=True,
            )
        except Exception as exc:
            outputs = [exc] * len(pending)

        for i, output in zip(pending, outputs):
            try:
                if isinstance(output, BaseException):
                    raise output
                text = response_text(output)
                values = parse(text)
                for j in range(2):
                    if records[i][j] is None or not records[i][j][1]:
                        records[i][j] = values[j]
                inputs[i]["feedback"] = (
                    f"Re-read the image and correct this draft: {text}\n"
                    f"Payment reconciles: {values[0][1]}. "
                    f"Items reconcile: {values[1][1]}. "
                    "Check each printed digit; never invent values to force agreement."
                )
            except Exception as exc:
                print(f"{images[i].name}: {type(exc).__name__}, status={getattr(exc, 'status_code', None)}")
                inputs[i]["feedback"] = "Previous attempt failed. Read the image again and return the full JSON."

    for path, row in zip(images, records):
        if any(x is None or not x[1] for x in row):
            print(f"Warning: {path.name} still has unverified amounts; inspect the test results.")

    return {
        query: (
            f"HK${sum((row[j][0] for row in records), zero):.2f}"
            if all(row[j] is not None for row in records)
            else "ERROR: receipt data could not be read"
        )
        for j, query in enumerate((QUERY_1, QUERY_2))
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
