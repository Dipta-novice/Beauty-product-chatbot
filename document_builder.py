import pandas as pd
from langchain_core.documents import Document

def build_documents(csv_path: str) -> list[Document]:
    """Build one rich document per product for retrieval."""
    df = pd.read_csv(csv_path)
    docs = []
    
    for _, row in df.iterrows():
        skin_types = [
            skin
            for skin in ["Combination", "Dry", "Normal", "Oily", "Sensitive"]
            if bool(row[skin])
        ]
        content = f"""🏷️ **{row['Name']} by {row['Brand']}**
💰 **Price:** ${row['Price']} | ⭐ **Rating:** {row['Rank']}/5
✅ **Skin Types:** 
  - Combination: {'✅' if row['Combination'] else '❌'}
  - Dry: {'✅' if row['Dry'] else '❌'}
  - Normal: {'✅' if row['Normal'] else '❌'}
  - Oily: {'✅' if row['Oily'] else '❌'}
  - Sensitive: {'✅' if row['Sensitive'] else '❌'}
🧪 **Ingredients:** {row['Ingredients']}
💧 **Benefits:** {row['Benefits']}
🎁 **Offers:** {row['Offers']}
🔄 **Returns:** {row['ReturnPolicy']}"""
        
        doc = Document(
            page_content=content.strip(),
            metadata={
                'name': str(row['Name']),
                'brand': str(row['Brand']),
                'product': str(row['Name']),
                'ingredients': str(row['Ingredients']),
                'category': str(row['Label']),
                'label': str(row['Label']),
                'benefits': str(row.get('Benefits', '')),
                'offers': str(row.get('Offers', '')),
                'return_policy': str(row.get('ReturnPolicy', '')),
                'price': float(row['Price']),
                'rank': float(row['Rank']),
                'skin_types': skin_types,
            }
        )
        docs.append(doc)
    
    print(f"Built {len(docs)} product documents")
    return docs
