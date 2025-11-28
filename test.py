from extractors.sec_10q import SECExtractor
import json

if __name__ == "__main__":
    extractor = SECExtractor(debug=True)
    sample_path = r"D:\Moiz\Projects\batch disclosure\data\sec\1227500_10-Q_2025-11-03.html"
    meta = {"doc_id": "1227500_2025-11-03", "date": "2025-11-03", "cik": 1227500}
    results = extractor.extract(sample_path, meta)
    print("JSON:")
    print(json.dumps([r.__dict__ for r in results], indent=2, default=str))