"""S4: PII masking before text reaches an external LLM."""
from src.pii import mask_pii, mask_pii_report


def test_email_ip_phone():
    m = mask_pii("Mail jane.doe@acme.de or call +49 30 1234567, server 10.0.0.12")
    assert "jane.doe@acme.de" not in m and "1234567" not in m and "10.0.0.12" not in m
    assert "[EMAIL]" in m and "[PHONE]" in m and "[IP]" in m


def test_card_and_iban():
    m, counts = mask_pii_report("Card 4111 1111 1111 1111, IBAN DE89 3704 0044 0532 0130 00")
    assert counts == {"CARD": 1, "IBAN": 1}
    assert "4111" not in m and "DE89" not in m


def test_non_luhn_long_number_is_not_a_card():
    assert "[CARD]" not in mask_pii("Serial 1234 5678 9012 3456")


def test_account_numbers_keep_label():
    m = mask_pii("Kundennummer: 4471-2290 and account no. 99887766")
    assert m == "Kundennummer: [ACCOUNT] and account no. [ACCOUNT]"


def test_leaves_ids_dates_versions_alone():
    s = "ticket TKT-00012 on 2024-03-15, version 2.14.0, at 10:30, port 8080, 3 users"
    assert mask_pii(s) == s


def test_url_credentials_and_ipv6():
    m = mask_pii("see https://bob:s3cret@git.example.com/repo and fe80::1ff:fe23:4567:890a")
    assert "s3cret" not in m and "git.example.com" in m and "fe80" not in m


def test_empty_and_non_string():
    assert mask_pii("") == ""
    assert mask_pii(None) == ""
    assert mask_pii(float("nan")) == ""
