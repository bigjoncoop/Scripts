# pip install -U google-generativeai pandas

import pandas as pd
import google.generativeai as genai
import json
import time

# --- CONFIGURATION ---
API_KEY = "YOUR_GOOGLE_AI_API_KEY"  # Replace with your actual key
MASTER_FILE = "amazon_products.csv"   # Your file with ASIN and Title
SOLD_FILE = "ebay_sold.csv"           # Your file with eBay titles
OUTPUT_FILE = "unsold_products.csv"
CHUNK_SIZE = 20  # How many products to send to the AI at once

# Setup Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

def get_sold_matches(master_chunk, sold_titles):
    """
    Asks Gemini to compare a chunk of master products against the sold list.
    Returns a list of ASINs that should be removed.
    """
    
    # Prepare the prompt
    master_list_str = "\n".join([f"ASIN: {row['ASIN']} | Title: {row['Title']}" for _, row in master_chunk.iterrows()])
    sold_list_str = "\n".join(sold_titles)

    prompt = f"""
    I have two lists of products. 
    List 1 is my Current Inventory (ASIN and Title).
    List 2 is a list of items I just Sold on eBay (Titles only).

    Your task: Identify which items from List 1 have been sold in List 2.
    Note: Titles will not match exactly. eBay titles may have extra keywords or different formatting, 
    but they refer to the same model, quantity, or specific product (e.g., "218A Toner 4 Pack" matches "LEMERO 218A Toner Cartridges 4 Pack").

    LIST 1 (Inventory):
    {master_list_str}

    LIST 2 (Sold on eBay):
    {sold_list_str}

    Return ONLY a JSON array of the ASIN numbers that are considered SOLD. 
    If no matches are found, return [].
    Format: ["ASIN1", "ASIN2"]
    """

    try:
        response = model.generate_content(prompt)
        # Clean the response to ensure it's valid JSON
        text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return []

def main():
    print("Loading files...")
    # Load Master CSV (Expected headers: ASIN, Title)
    df_master = pd.read_csv(MASTER_FILE)
    # Load Sold CSV (Expected header: Title)
    df_sold = pd.read_csv(SOLD_FILE)
    
    # Clean empty rows
    df_master = df_master.dropna(subset=['ASIN', 'Title'])
    sold_titles = df_sold['Title'].dropna().tolist()

    all_sold_asins = []

    print(f"Comparing {len(df_master)} inventory items against {len(sold_titles)} sold items...")

    # Process in chunks to avoid token limits and improve accuracy
    for i in range(0, len(df_master), CHUNK_SIZE):
        chunk = df_master.iloc[i:i + CHUNK_SIZE]
        print(f"Processing items {i} to {i + len(chunk)}...")
        
        sold_asins = get_sold_matches(chunk, sold_titles)
        all_sold_asins.extend(sold_asins)
        
        # Small delay to respect rate limits if using free tier
        time.sleep(2)

    # Remove duplicates from our "sold" list
    all_sold_asins = list(set(all_sold_asins))
    
    print(f"AI identified {len(all_sold_asins)} items as sold.")

    # Filter the master dataframe
    final_df = df_master[~df_master['ASIN'].isin(all_sold_asins)]

    # Save results
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! {len(final_df)} unsold items saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()