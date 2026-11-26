import pytest

from complaint_router.labels import (
    DESCRIPTIONS_EN,
    EXCLUDED_PRODUCTS,
    LABELS,
    PRODUCT_TO_LABEL,
    TEAMS_FR,
    label_for_product,
)
from complaint_router.text import clean_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("On XX/XX/XXXX I called XXXX XXXX bank", "On XXXX I called XXXX bank"),
        ("They charged {$1200.00} and {$35.00}", "They charged $1200.00 and $35.00"),
        ("  too\n\nmany   spaces\t", "too many spaces"),
        ("XXXX, XXXX. XXXX the agent", "XXXX the agent"),
        ("Normal text without masks", "Normal text without masks"),
    ],
)
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


def test_clean_text_does_not_touch_words_containing_x():
    assert clean_text("Max paid by XXXX via Xbox") == "Max paid by XXXX via Xbox"


def test_clean_text_is_idempotent():
    # L'API nettoie un texte qui peut l'avoir déjà été : le résultat ne doit pas changer
    raw = "On XX/XX/XXXX, XXXX charged me {$40.00}.  Then XXXX XXXX called."
    assert clean_text(clean_text(raw)) == clean_text(raw)


def test_every_label_has_a_team_and_a_description():
    assert set(TEAMS_FR) == set(LABELS)
    assert set(DESCRIPTIONS_EN) == set(LABELS)


def test_product_mapping_targets_known_labels():
    assert set(PRODUCT_TO_LABEL.values()) == set(LABELS)


def test_renamed_products_share_a_label():
    old = "Credit reporting, credit repair services, or other personal consumer reports"
    new = "Credit reporting or other personal consumer reports"
    assert label_for_product(old) == label_for_product(new) == "credit_reporting"
    assert label_for_product("Credit card or prepaid card") == label_for_product("Credit card")


def test_excluded_and_unknown_products_are_dropped():
    for product in EXCLUDED_PRODUCTS:
        assert label_for_product(product) is None
    assert label_for_product("Something new") is None
    assert label_for_product(None) is None
