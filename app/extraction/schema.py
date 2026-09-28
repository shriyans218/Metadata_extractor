"""
Field schema for the metadata extraction task.

Centralizing field names + descriptions here means the prompt,
the evaluator, and the API response all stay in sync -- add a new
field once, and every layer of the app picks it up.
"""

from __future__ import annotations

# Canonical field names used everywhere in the app (Python-friendly keys).
FIELDS = [
    "agreement_value",
    "agreement_start_date",
    "agreement_end_date",
    "renewal_notice_days",
    "party_one",
    "party_two",
]

# Human-readable descriptions fed into the LLM prompt. Kept semantic
# (not keyword lists) so the model reasons about *meaning*, which is
# what lets this generalize across differently worded contracts.
FIELD_DESCRIPTIONS = {
    "agreement_value": (
        "The monetary rent amount agreed upon in the contract "
        "(e.g. monthly rent value). Return only the numeric amount, "
        "no currency symbol, no commas. Use the figure written in "
        "numerals; if the numerals and the amount in words disagree, "
        "still return the numerals."
    ),
    "agreement_start_date": (
        "The date the rental/lease agreement begins or was executed. "
        "Return in DD.MM.YYYY format."
    ),
    "agreement_end_date": (
        "The date the rental/lease agreement ends or expires. "
        "If no end date is written but the start date and a duration "
        "are (e.g. '11 months', '1 year'), compute the end date as the "
        "last day of that period (start date + duration - 1 day). "
        "Use null only if neither an end date nor a duration is given. "
        "Return in DD.MM.YYYY format."
    ),
    "renewal_notice_days": (
        "The number of days' advance notice required to terminate, "
        "vacate or renew the agreement. Consider every clause that "
        "mentions a notice period, not only one titled 'renewal'. "
        "If the notice is written in months, convert to days "
        "(one month = 30 days). Return only the integer number of days."
    ),
    "party_one": (
        "The first party to the agreement -- typically the lessor / "
        "owner / landlord of the property. Return the name only, without "
        "honorific titles such as Mr., Mrs., Sri, Shri or Smt."
    ),
    "party_two": (
        "The second party to the agreement -- typically the lessee / "
        "tenant renting the property. Return the name only, without "
        "honorific titles such as Mr., Mrs., Sri, Shri or Smt."
    ),
}

# Maps our canonical keys to the exact column names used in the
# provided train.csv / test.csv, so evaluation can join cleanly.
CSV_COLUMN_MAP = {
    "agreement_value": "Aggrement Value",
    "agreement_start_date": "Aggrement Start Date",
    "agreement_end_date": "Aggrement End Date",
    "renewal_notice_days": "Renewal Notice (Days)",
    "party_one": "Party One",
    "party_two": "Party Two",
}
