# -------------------------------
# Install required packages
# -------------------------------
!pip install -q sentence-transformers transformers faiss-cpu fastapi uvicorn pyngrok
from fastapi.middleware.cors import CORSMiddleware

# -------------------------------
# Imports
# -------------------------------
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from fastapi import FastAPI, Request
from pydantic import BaseModel
import nest_asyncio
import uvicorn
from pyngrok import ngrok

# -------------------------------
# STEP 1: Load menu
# -------------------------------
menu_df = pd.read_csv("menu_items_full.csv")
print(f"✅ Loaded menu with {len(menu_df)} items")

# -------------------------------
# STEP 2: Build embeddings
# -------------------------------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
menu_texts = menu_df["item_name_en"].tolist()
menu_embeddings = embed_model.encode(menu_texts, convert_to_numpy=True)

# Build FAISS index
dimension = menu_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(menu_embeddings)
print("✅ FAISS index built")

# -------------------------------
# STEP 3: Setup Flan-T5 pipeline
# -------------------------------
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=200)

# -------------------------------
# STEP 4: Conversation memory
# -------------------------------
conversation_history = []
last_retrieved_item = None

# -------------------------------
# STEP 5: Helper functions
# -------------------------------
def normalize_query(query):
    return query.lower().strip()

def retrieve(query, top_k=5):
    """Return top-k menu items for a query"""
    q_vec = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(q_vec, top_k)
    retrieved = menu_df.iloc[I[0]]
    return retrieved

def format_context(retrieved_df, starred_item):
    """Format menu context and mark the last retrieved item with ★"""
    context_lines = []
    for _, row in retrieved_df.iterrows():
        star = "★" if row["item_name_en"] == starred_item else ""
        line = f"{row['item_name_en']} {star}| Price: {row['price']} {row['currency_iso']} | Calories: {row['calories']}"
        context_lines.append(line)
    return "\n".join(context_lines)

# -------------------------------
# STEP 6: RAG query function (UNCHANGED)
# -------------------------------
def rag_query(user_query):
    global conversation_history, last_retrieved_item

    user_query_norm = normalize_query(user_query)

    # Retrieve top-k items
    retrieved = retrieve(user_query_norm, top_k=5)

    # --- Print retrieved items ---
    print(f"\n🔹 Retrieved items for query: {user_query}")
    for idx, row in retrieved.iterrows():
        print(f"{idx+1}. {row['item_name_en']} | Price: {row['price']} | Calories: {row['calories']}")

    # Update last_retrieved_item if user mentions it explicitly
    for item in retrieved["item_name_en"]:
        if item.lower() in user_query_norm:
            last_retrieved_item = item
            break

    # If no explicit mention, keep last_retrieved_item as is
    if last_retrieved_item is None and len(retrieved) > 0:
        last_retrieved_item = retrieved.iloc[0]["item_name_en"]

    print(f"🟢 Last retrieved item: {last_retrieved_item}")

    # --- ONLY include the last retrieved item in the context ---
    last_item_row = menu_df[menu_df["item_name_en"] == last_retrieved_item]
    retrieved_combined = pd.concat([last_item_row, retrieved]).drop_duplicates().reset_index(drop=True)
    context = format_context(retrieved_combined, last_retrieved_item)

    # Add conversation history (last 3 turns)
    history_text = "\n".join([f"{t['role'].capitalize()}: {t['content']}" for t in conversation_history[-3:]])

    # Prompt LLM
    prompt = f"""
You are a restaurant assistant. Use ONLY the menu context below to answer naturally.
Rules:
1. Only provide information about the item marked with ★ (the last retrieved item: {last_retrieved_item}).
2. Always mention the item's name, price, and calories.
3. If the user query refers to 'it', 'this item', 'its', or similar, it refers to the ★ item.
4. If the query does not match the last retrieved item, respond with "Item not found".
5. Your answers should be short, precise, and directly answer the user's question.

Menu Context:
{context}

Conversation:
{history_text}

Question: {user_query}
Answer:
"""
    output = pipe(prompt)[0]["generated_text"]
    answer = output.strip()

    conversation_history.append({"role":"user","content":user_query})
    conversation_history.append({"role":"assistant","content":answer})

    return answer

# -------------------------------
# STEP 7: FastAPI setup
# -------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],   # allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query(req: QueryRequest):
    answer = rag_query(req.question)
    return {"answer": answer}

# -------------------------------
# STEP 8: Run FastAPI with ngrok
# -------------------------------
NGROK_AUTH_TOKEN = "32hXYSJAcYGSC448egK9RbkHlkI_6giQwPahGgWKeG34ZvXeB"  # <-- replace with your token
!ngrok config add-authtoken $NGROK_AUTH_TOKEN

# Open ngrok tunnel
public_url = ngrok.connect(8000)
print("🚀 Public URL:", public_url)

# Needed for Colab
nest_asyncio.apply()

# Run server
uvicorn.run(app, host="0.0.0.0", port=8000)
