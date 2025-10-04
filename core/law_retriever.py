import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from typing import List, Dict

class LawRetriever:
    def __init__(self, model_name: str, openai_api_key: str):
        print("Initializing Law Retriever...")
        self.DATA_PATH = os.path.join("data", "laws.xlsx")
        self.EMBEDDINGS_PATH = os.path.join("data", "Embedding.npy")
        
        # Load embedding model from Hugging Face Hub
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name)

        # Setup OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)

        # Load legal data and embeddings
        self.df_laws, self.law_embeddings = self._load_data_and_embeddings()
        self.law_texts_list = self.df_laws['text'].tolist()
        print("✅ Law Retriever initialized successfully.")

    def _load_data_and_embeddings(self):
        """Loads law data and embeddings, creating embeddings if they don't exist."""
        if not os.path.exists(self.DATA_PATH):
            raise FileNotFoundError(f"Law data file not found at {self.DATA_PATH}.")
        
        df = pd.read_excel(self.DATA_PATH)
        
        if os.path.exists(self.EMBEDDINGS_PATH):
            print(f"Loading existing embeddings from {self.EMBEDDINGS_PATH}")
            embeddings = np.load(self.EMBEDDINGS_PATH)
            if len(df) != embeddings.shape[0]:
                print("⚠️ Mismatch detected. Regenerating embeddings.")
                embeddings = self._generate_and_save_embeddings(df)
        else:
            print("No embeddings file found. Generating new ones...")
            embeddings = self._generate_and_save_embeddings(df)
            
        return df, embeddings

    def _generate_and_save_embeddings(self, df):
        """Generates embeddings for the law texts and saves them to a file."""
        texts = df['text'].tolist()
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True, batch_size=128)
        print(f"Saving new embeddings to {self.EMBEDDINGS_PATH}")
        np.save(self.EMBEDDINGS_PATH, embeddings)
        return embeddings

    def search_and_answer(self, query: str, chat_history: List[Dict]) -> Dict:
        """
        Performs the advanced RAG pipeline: retrieve, re-rank, and generate.
        """
        print(f"Executing search for query: '{query}'")
        
        # --- 1. Initial Retrieval ---
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.law_embeddings)
        # Get top 5 initial candidates
        top_indices = np.argsort(similarities[0])[::-1][:5]
        newly_fetched_sources = [{'law_text': self.law_texts_list[int(i)], 'source_index': int(i)} for i in top_indices]
        
        # --- 2. Combine with historical sources for re-ranking ---
        existing_sources = []
        for msg in chat_history:
            if msg.get('role') == 'assistant' and 'sources' in msg:
                existing_sources.extend(msg['sources'])
        
        # Create a unique set of all candidate sources (new + old)
        combined_sources_map = {src['law_text']: src for src in existing_sources}
        for src in newly_fetched_sources:
            if src['law_text'] not in combined_sources_map:
                combined_sources_map[src['law_text']] = src
        
        all_candidate_sources = list(combined_sources_map.values())

        # --- 3. Re-ranking Logic ---
        final_sources = []
        if all_candidate_sources:
            print(f"Re-ranking from {len(all_candidate_sources)} unique candidate sources.")
            all_candidate_texts = [src['law_text'] for src in all_candidate_sources]
            all_candidate_embeddings = self.embedding_model.encode(all_candidate_texts)
            
            reranking_similarities = cosine_similarity(query_embedding, all_candidate_embeddings)
            # Get top 7 most relevant sources after re-ranking
            reranked_indices = np.argsort(reranking_similarities[0])[::-1][:7]
            final_sources = [all_candidate_sources[i] for i in reranked_indices]

        # --- 4. Generation ---
        context_for_gpt = "".join([f"المصدر رقم [{i+1}]:\n{src['law_text']}\n\n" for i, src in enumerate(final_sources)])
        
        history_for_gpt = [{'role': msg['role'], 'content': msg['content']} for msg in chat_history]
        
        system_prompt = (
            "أنت مساعد قانوني خبير ومختص في القوانين السعودية. مهمتك هي الإجابة على سؤال المستخدم الأخير بدقة ووضوح، "
            "معتمداً **حصرياً** على نصوص المواد القانونية التي أزودك بها كمصادر وسياق المحادثة السابق. "
            "إذا لم تكن الإجابة موجودة بشكل واضح وصريح ضمن المصادر المقدمة، أجب بـ: "
            "'لا أجد إجابة واضحة في المصادر المتوفرة لدي بخصوص هذا السؤال.' "
            "لا تحاول أبداً استنتاج أو تخمين الإجابة. اذكر دائماً أرقام المصادر التي استخدمتها في إجابتك، مثال: [المصدر 1]."
        )
        messages = [{"role": "system", "content": system_prompt}] + history_for_gpt
        user_prompt_with_context = f"بناءً على المصادر التالية، أجب على السؤال.\n\n## المصادر:\n{context_for_gpt}\n\n## السؤال:\n{query}"
        messages.append({"role": "user", "content": user_prompt_with_context})
        
        response = self.openai_client.chat.completions.create(model="gpt-5", messages=messages, temperature=0.1)
        gpt_answer = response.choices[0].message.content

        return {"answer": gpt_answer, "sources": final_sources}

