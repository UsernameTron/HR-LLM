"""Mock data for testing SEC EDGAR integration."""

MOCK_SUBMISSION_DATA = {
    "cik": "0000320193",
    "entityType": "operating",
    "sic": "3571",
    "sicDescription": "ELECTRONIC COMPUTERS",
    "name": "APPLE INC",
    "filings": {
        "recent": [
            {
                "accessionNumber": "0000320193-23-000077",
                "filingDate": "2023-11-03",
                "reportDate": "2023-09-30",
                "form": "10-K",
                "primaryDocument": "primary-doc.htm",
                "primaryDocDescription": "Annual report [Section 13 and 15(d), not S-K Item 405]"
            }
        ]
    }
}

MOCK_FILING_DATA = """
APPLE INC.
FORM 10-K
For the fiscal year ended September 30, 2023

Item 1. Business
Apple Inc. is a global technology company that designs, manufactures, and sells smartphones, personal computers, tablets, wearables and accessories.

Item 1A. Risk Factors
The Company's success depends on its ability to attract, develop, motivate and retain a highly skilled workforce.

Item 2. Properties
The Company's headquarters are located in Cupertino, California.

Employees
As of September 30, 2023, the Company had approximately 165,000 full-time equivalent employees. The Company believes its employee relations are good.
The Company continues to focus on growing its workforce and investing in talent acquisition to support its expanding operations.
The Company has implemented various initiatives to attract and retain top talent in a competitive market.

Human Capital Resources
The Company is committed to hiring and developing a diverse workforce. In fiscal 2023, we expanded our recruitment efforts
and increased our investment in employee development programs. Our growing workforce reflects our commitment to innovation
and expansion across key markets. We have seen positive results from our talent acquisition initiatives and continue to
maintain strong employee engagement levels."""
