import pandas as pd
import random

BENEFITS_MAP = {
    "Moisturizer": ["deep hydration", "anti-aging", "skin barrier repair", "SPF protection"],
    "Cleanser":    ["pore cleansing", "removes impurities", "balances pH", "gentle exfoliation"],
    "Treatment":   ["brightening", "acne control", "pigmentation reduction", "serum absorption"],
    "Eye cream":   ["reduces puffiness", "dark circle treatment", "fine line smoothing"],
    "Sun protect": ["broad spectrum SPF", "blue light defense", "water resistant"],
    "Face Mask":   ["deep cleanse", "instant glow", "detoxifying", "nourishing"],
    "Toner":       ["pore tightening", "pH balancing", "hydration boost"],
}

OFFERS = [
    "Buy 2 Get 1 Free", "15% off on first order", "Free shipping above ₹999",
    "No active offer", "Combo deal available", "Loyalty points 2x this week",
]

RETURN_POLICIES = [
    "7-day easy return",
    "No returns on opened products",
    "30-day return with original packaging",
    "Exchange only within 15 days",
]

def enrich_data(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip().str.lstrip('\ufeff')  # clean BOM

    def get_benefits(label):
        pool = BENEFITS_MAP.get(label, ["general skincare benefit"])
        return ", ".join(random.sample(pool, min(2, len(pool))))

    df["Benefits"]     = df["Label"].apply(get_benefits)
    df["Offers"]       = [random.choice(OFFERS) for _ in range(len(df))]
    df["ReturnPolicy"] = [random.choice(RETURN_POLICIES) for _ in range(len(df))]

    df.to_csv(output_path, index=False)
    print(f"Enriched CSV saved → {output_path}  ({len(df)} rows)")
    return df

if __name__ == "__main__":
    enrich_data("cosmetics.csv", "cosmetics_enriched.csv")