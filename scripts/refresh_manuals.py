import os
import sys

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ingest_from_s3 import ingest_from_s3

def main():
    manuals = [
        {"key": "Mahindra XUV 400 Owner’s Manual.pdf", "model": "Mahindra XUV 400"},
        {"key": "nexon-ev-owners-manual.pdf", "model": "Tata Nexon EV"}
    ]
    
    print("Starting full manual refresh...")
    
    for i, entry in enumerate(manuals):
        manual = entry["key"]
        model = entry["model"]
        print(f"\n--- Processing Manual {i+1}/{len(manuals)}: {manual} ({model}) ---")
        # Clear data only on the first manual
        clear = (i == 0)
        try:
            ingest_from_s3(manual, model, clear_existing=clear)
        except Exception as e:
            print(f"Error processing {manual}: {e}")
            # Depending on requirements, we might want to continue or stop
            # For this task, we stop on error to avoid partial state
            sys.exit(1)

    print("\nAll manuals refreshed successfully!")

if __name__ == "__main__":
    main()
