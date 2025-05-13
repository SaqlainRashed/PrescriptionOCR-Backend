try:
    import easyocr
    import pandas as pd
    import numpy as np
    import cv2
    import re
    import logging
    from typing import List, Dict, Optional, Tuple
    from pathlib import Path
    from fuzzywuzzy import process, fuzz
    import Levenshtein
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    import torch

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
except ImportError as e:
    raise ImportError(f"Missing required module: {e.name}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MedicineService:
    def __init__(self, db_path: str, model_dir: str = 'models'):
        if not Path(db_path).is_file():
            raise FileNotFoundError(f"Database file not found: {db_path}")

        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        self.reader = easyocr.Reader(
            ['en'],
            gpu=False,
            download_enabled=not any(model_path.glob("*.pth")),
            model_storage_directory=str(model_path),
            verbose=True
        )

        self.medicine_db = self._load_database(db_path)


    def _load_database(self, db_path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(db_path)
            df.columns = df.columns.str.strip().str.lower()

            expected_cols = ['medicine_name', 'short_composition1', 'pack_size_label',
                            'manufacturer_name', 'type', 'price(₹)']
            missing_cols = [col for col in expected_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing expected columns: {missing_cols}")

            text_cols = ['medicine_name', 'short_composition1', 'pack_size_label',
                            'manufacturer_name', 'type']
            for col in text_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.lower()

            df.dropna(subset=['medicine_name'], inplace=True)
            df.reset_index(drop=True, inplace=True)

            logger.info(f"Loaded medicine database with {len(df)} entries")
            return df
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            raise

    def extract_best_text(self, image_path: str) -> str:
        try:
            text = self.extract_handwritten_text(image_path)
            if not text or len(text.strip()) < 5 or "2nd generation" in text.lower():
                raise ValueError("TrOCR failed or gave unreliable text")
        except Exception as e:
            logger.warning(f"TrOCR failed: {e}. Falling back to EasyOCR.")
            text = self.extract_text(image_path)

        return text


    
    def extract_handwritten_text(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return extracted_text.lower()

    def extract_text(self, image_path: str) -> str:
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

            result = self.reader.readtext(img, detail=0)

            # Join with newline instead of space
            extracted_text = '\n'.join(result).lower()
            extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()

            logger.info(f"Extracted text (lines): {extracted_text[:100]}...")
            return extracted_text

        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise


    def extract_medicine_entries(self, text: str) -> List[str]:
        # Normalize bullets and numbering
        lines = re.split(r'(?:\n+|\r+|\d+[^a-zA-Z0-9]\s*)', text)
        blacklist = {"take", "daily", "before", "after", "tablet", "mg", "ml"}
        
        return [
            entry.strip().lower()
            for entry in lines
            if entry.strip() and any(c.isalpha() for c in entry)
            and not all(word in entry.lower() for word in blacklist)
        ]



    def process_prescription(self, image_path: str, mode: str = "SINGLE_MED") -> Tuple[str, List[Dict]]:
        """Full pipeline with multi-medicine support"""
        extracted_text = self.extract_text(image_path)
        logger.info(f"Extracted text: {extracted_text}")
        logger.info(f"Processing mode: {mode}")
        
        if mode == "MULTIPLE_MEDS":
            medicine_entries = self.extract_medicine_entries(extracted_text)
            logger.info(f"Extracted {len(medicine_entries)} medicine entries")
            all_suggestions = []
            
            for entry in medicine_entries:
                suggestions = self.suggest_medicines(entry)
                if suggestions:
                    # Add entry context to suggestions
                    for suggestion in suggestions:
                        suggestion['prescription_entry'] = entry
                    all_suggestions.extend(suggestions)
            
            return extracted_text, all_suggestions
        else:
            # Original single-medicine processing
            suggestions = self.suggest_medicines(extracted_text)
            return extracted_text, suggestions

    def suggest_medicines(self, extracted_text: str, top_n: int = 5, threshold: int = 0) -> List[Dict]:
        try:
            medicines = self.medicine_db['medicine_name'].tolist()
            matches = process.extract(
                extracted_text.lower(),
                medicines,
                scorer=fuzz.token_set_ratio,
                limit=top_n
            )

            suggestions = []
            for match_name, score in matches:
                if score < threshold:
                    continue

                rows = self.medicine_db[self.medicine_db['medicine_name'] == match_name]
                for _, row in rows.iterrows():
                    suggestions.append({
                        'medicine_name': row.get('medicine_name', ''),
                        'short_composition1': row.get('short_composition1', ''),
                        'pack_size_label': row.get('pack_size_label', ''),
                        'manufacturer_name': row.get('manufacturer_name', ''),
                        'type': row.get('type', ''),
                        'price(₹)': row.get('price(₹)', ''),
                        'confidence_score': score
                    })
            return suggestions

        except Exception as e:
            logger.error(f"Medicine suggestion failed: {e}")
            return []

    # def process_prescription(self, image_path: str, mode: str = "SINGLE_MED") -> Tuple[str, List[Dict]]:
    #     extracted_text = self.extract_text(image_path)
    #     all_suggestions = []

    #     if mode.upper() == "MULTIPLE_MEDS":
    #         medicine_entries = self.extract_medicine_entries(extracted_text)
    #         for entry in medicine_entries:
    #             suggestions = self.suggest_medicines(entry)
    #             for suggestion in suggestions:
    #                 suggestion['prescription_entry'] = entry
    #             all_suggestions.extend(suggestions)
    #     else:
    #         all_suggestions = self.suggest_medicines(extracted_text)

    #     logger.info(f"Total suggestions returned: {len(all_suggestions)}")
    #     return extracted_text, all_suggestions
