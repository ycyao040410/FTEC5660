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
        max_tokens=4096,
        timeout=60,
        max_retries=1,
    )
    task = """
Transcribe one supermarket receipt. Return JSON only:
{{"payment":null,"subtotal":null,"rounding":null,
"charges":[],"discounts":[]}}
Use decimal strings for money. Each charges/discounts entry must be:
{{"label":"short printed label","amount":"decimal string"}}
payment: actual final payment after ROUNDING, not cash tendered,
change, card balance, or a duplicated payment record.
subtotal: printed SUBTOTAL / 小計 before ROUNDING.
rounding: printed signed ROUNDING adjustment; "0.00" if absent.
charges: every non-discount transaction line contributing to SUBTOTAL,
including goods, plastic bags and other fees. Preserve printed signs.
discounts: every applied discount, promotion, coupon, member/app saving,
MB PRICE, packaging reduction and percentage-off line, as positive amounts.
Copy the rightmost posted line amounts exactly. Promotional wording may
contain a DIFFERENT amount; use the actual posted amount, not that wording.
Extended line totals already include quantity: do not multiply them again.
Keep repeated transaction lines as separate entries. Do not deduplicate.
Exclude SUBTOTAL, ROUNDING, payments, change, balances, points and savings
summaries from both lists. Never count a discount twice.
Do not recalculate percentage discounts or invent unprinted discounts.
Use null for unreadable money; never replace unreadable values with zero.
Transcribe what is printed; never adjust amounts to force an equation.
Reading strategy: {strategy}
Diagnostic history, if supplied (may contain mistakes):
{history}
""".strip()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Accurately transcribe English and Chinese receipts."),
        ("human", [
            {"type": "text", "text": task},
            {"type": "image_url", "image_url": {"url": "{image_url}"}},
        ]),
    ])
    return prompt | model | StrOutputParser()


def answer_queries(chain: Any, images: list[Path]) -> dict[str, Any]:
    from collections import Counter
    zero = Decimal("0.00")
    cent = Decimal("0.01")
    def money(value):
        text = str(value).strip()
        if not re.fullmatch(r"-?\d+(?:\.\d{1,2})?", text):
            raise ValueError("Missing or invalid decimal amount")
        return Decimal(text).quantize(cent)
    def amounts(data, field):
        rows = data[field]
        if not isinstance(rows, list):
            raise ValueError(f"{field} must be a list")
        values = [money(row["amount"]) for row in rows]
        if field == "discounts":
            values = [abs(value) for value in values]
        return tuple(sorted(value for value in values if value != zero))
    def consensus(votes):
        ranked = votes.most_common(2)
        if ranked and ranked[0][1] >= 2:
            if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                return ranked[0][0]
        return None
    states = []
    for path in images:
        state = {
            "path": path,
            "url": None,
            "paid": Counter(),
            "full": Counter(),
            "history": [],
        }
        try:
            state["url"] = image_data_url(path)
        except Exception as exc:
            print(f"Cannot open {path.name}: {type(exc).__name__}")
        states.append(state)
    strategies = [
        "Read from top to bottom, recording every posted transaction line.",
        "Read independently from bottom to top. Check every amount and sign.",
        "Read independently. Match each amount to its printed row; check "
        "bag fees, repeated rows, promotions and faint decimal digits.",
        "Audit the image using the diagnostic history. Resolve omissions "
        "and disagreements from printed evidence, not from candidate votes.",
        "Make a fresh independent transcription. Recheck the amount column "
        "digit by digit, keeping every applied discount and charge.",
    ]
    for attempt, strategy in enumerate(strategies):
        pending = [
            state for state in states
            if state["url"] is not None
            and (
                consensus(state["paid"]) is None
                or consensus(state["full"]) is None
            )
        ]
        if not pending:
            break
        inputs = [
            {
                "image_url": state["url"],
                "strategy": strategy,
                "history": (
                    json.dumps(state["history"][-3:], ensure_ascii=False)
                    if attempt == 3 else "No previous transcription supplied."
                ),
            }
            for state in pending
        ]
        try:
            outputs = chain.batch(
                inputs,
                config={"max_concurrency": 3},
                return_exceptions=True,
            )
        except Exception as exc:
            outputs = [exc] * len(pending)
        for state, output in zip(pending, outputs):
            errors = []
            if isinstance(output, BaseException):
                text = f"API request failed: {type(output).__name__}"
                data = None
                errors.append(text)
            else:
                text = response_text(output)
                try:
                    data = json.loads(text[text.index("{"):text.rindex("}") + 1])
                    if not isinstance(data, dict):
                        raise ValueError("Expected a JSON object")
                except Exception:
                    data = None
                    errors.append("Invalid JSON object")
            if data is not None:
                try:
                    subtotal = money(data["subtotal"])
                    rounding = money(data["rounding"])
                    paid = subtotal + rounding
                    if data.get("payment") is not None:
                        if money(data["payment"]) != paid:
                            raise ValueError("Payment != SUBTOTAL + ROUNDING")
                    state["paid"][(paid, subtotal, rounding)] += 1
                except Exception as exc:
                    errors.append(f"Payment check: {exc}")
                try:
                    subtotal = money(data["subtotal"])
                    charges = amounts(data, "charges")
                    discounts = amounts(data, "discounts")
                    difference = sum(charges, zero) - sum(discounts, zero) - subtotal
                    if difference != zero:
                        raise ValueError(f"Charges - discounts - SUBTOTAL = {difference}")
                    state["full"][(subtotal, charges, discounts)] += 1
                except Exception as exc:
                    errors.append(f"Ledger check: {exc}")
            state["history"].append({
                "transcription": text[:12000],
                "validation_errors": errors,
            })
    total_paid = zero
    total_full = zero
    paid_ok = full_ok = True
    for state in states:
        paid = consensus(state["paid"])
        full = consensus(state["full"])
        if paid is None:
            paid_ok = False
        else:
            total_paid += paid[0]
        if full is None:
            full_ok = False
        else:
            total_full += full[0] + sum(full[2], zero)
        paid_text = f"HK${paid[0]:.2f}" if paid is not None else "UNRESOLVED"
        full_text = (
            f"HK${full[0] + sum(full[2], zero):.2f}"
            if full is not None else "UNRESOLVED"
        )
        print(f"{state['path'].name}: paid={paid_text}, without_discount={full_text}")
    return {
        QUERY_1: f"HK${total_paid:.2f}" if paid_ok else "ERROR: payment unresolved",
        QUERY_2: f"HK${total_full:.2f}" if full_ok else "ERROR: receipt ledger unresolved",
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
