import os
import torch
import whisper
import sounddevice as sd
import numpy as np
import queue
import sys
import json
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# ---------------- CONFIG (OPTIMIZED) ----------------
OUTPUT_DIR = "stablelm2_finetuned_single_template"  # your trained model path
TEMPLATE_FILE = "template.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"
TOP_K = 3
SIM_THRESHOLD = 0.80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
CHANNELS = 1
# --- SPEED INCREASE 1: Reduced recording time ---
DURATION = 15  # Reduced from 15 to 5 seconds for much faster turns

# ---------------- TEMPLATE MANAGER (OPTIMIZED) ----------------
class TemplateManager:
    def __init__(self, template_file_path):
        with open(template_file_path, "r") as f:
            self.templates = json.load(f)
        self.embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE}
        )
        self.template_vectors = {}
        for emotion, subcats in self.templates.items():
            self.template_vectors[emotion] = {}
            if emotion == "default":
                # --- SPEED INCREASE 2: Batch embedding is much faster ---
                self.template_vectors["default"] = self._embed_texts_batch([t["text"] for t in subcats])
                continue
            for sub, templates in subcats.items():
                self.template_vectors[emotion][sub] = self._embed_texts_batch([t["text"] for t in templates])

    def _embed_texts_batch(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts in a single, efficient batch."""
        if not texts:
            return np.array([])
        embeddings = self.embedder.embed_documents(texts)
        return np.array(embeddings)

    def find_best_template(self, emotion, user_text):
        emotion_bucket = self.templates.get(emotion, None)
        vectors = None
        templates = None
        if not emotion_bucket:
            # fallback to default
            templates = self.templates.get("default", [])
            if not templates: return None
            vectors = self.template_vectors.get("default")
        else:
            # match inside emotion
            best_score = -1
            best_response = None
            if not user_text: return None
            user_vec = np.array(self.embedder.embed_query(user_text)).reshape(1, -1)
            for sub_category, templates_list in emotion_bucket.items():
                vectors_subset = self.template_vectors.get(emotion, {}).get(sub_category)
                if vectors_subset is None or len(vectors_subset) == 0: continue
                
                sims = cosine_similarity(user_vec, vectors_subset)[0]
                max_idx = np.argmax(sims)
                if sims[max_idx] > best_score:
                    best_score = sims[max_idx]
                    best_response = templates_list[max_idx]
            if best_score >= SIM_THRESHOLD:
                return best_response
            else:
                return None
        # fallback default search
        if vectors is not None and templates is not None and len(vectors) > 0:
            user_vec = np.array(self.embedder.embed_query(user_text)).reshape(1, -1)
            sims = cosine_similarity(user_vec, vectors)[0]
            max_idx = np.argmax(sims)
            if sims[max_idx] >= SIM_THRESHOLD:
                return templates[max_idx]
        return None

# ---------------- CONVERSATION MEMORY ----------------
def _docs_from_turns(turns: List[dict]):
    return [Document(page_content=t["content"], metadata={"role": t["role"]}) for t in turns]

def _add_turn_to_store(turn, vector_store_obj, embedder_obj):
    docs = _docs_from_turns([turn])
    if vector_store_obj is None:
        return FAISS.from_documents(docs, embedder_obj)
    else:
        vector_store_obj.add_documents(docs)
        return vector_store_obj

def _retrieve_relevant(user_input: str, vector_store_obj, k=TOP_K):
    if vector_store_obj is None: return []
    retriever = vector_store_obj.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(user_input)

def _build_contexted_prompt(retrieved_docs, emotion, user_input):
    sys_prompt = "You are a compassionate mental health counseling assistant. Answer carefully and helpfully."
    prompt = f"{sys_prompt}\n\n"
    if retrieved_docs:
        prompt += "Relevant past turns:\n"
        for d in retrieved_docs:
            role = d.metadata.get("role", "user")
            prompt += f"{role.capitalize()}: {d.page_content}\n"
        prompt += "\n"
    prompt += f"User (Emotion: {emotion}): {user_input}\nAssistant:"
    return prompt

class ChatModule:
    def __init__(self):
        print("Loading chat model from:", OUTPUT_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            OUTPUT_DIR,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(DEVICE)
        self.model.eval()

        self.embedder = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE}
        )
        
        if os.path.exists(TEMPLATE_FILE):
            self.template_manager = TemplateManager(TEMPLATE_FILE)
        else:
            self.template_manager = None
            print(f"Warning: '{TEMPLATE_FILE}' not found. Template matching will be disabled.")
        
        # --- SPEED INCREASE 3: Use a smaller, faster Whisper model ---
        # Switch to "base" or "tiny" for speed. "small" is more accurate but slower.
        self.whisper_model = whisper.load_model("base") 
        self.conversation_history = []
        self.vector_store_local = None

    def _record_audio(self, duration=DURATION, samplerate=SAMPLE_RATE):
        print(f"\n🎤 Recording for {duration} seconds... Speak now!")
        q = queue.Queue()

        def callback(indata, frames, time, status):
            if status: print(status, file=sys.stderr)
            q.put(indata.copy())

        audio_chunks = []
        with sd.InputStream(samplerate=samplerate, channels=CHANNELS, callback=callback):
            for _ in range(int(duration * samplerate / 1024)):
                 audio_chunks.append(q.get())
        
        if not audio_chunks: return np.array([], dtype=np.float32)
        
        audio = np.concatenate(audio_chunks, axis=0)
        print("✅ Recording finished.")
        return audio.flatten().astype(np.float32)
    
    def speech_to_text_from_data(self, audio_data: np.ndarray):
        if audio_data is None or audio_data.size == 0:
            return ""
        print("🧠 Transcribing with Whisper...")
        result = self.whisper_model.transcribe(audio_data, fp16=(DEVICE=="cuda"))
        text = result["text"].strip()
        print(f"\n[You said] {text}")
        return text

    def get_response(self, emotion, user_input):
        response_text = ""
        best_template = None
        if self.template_manager:
            best_template = self.template_manager.find_best_template(emotion, user_input)
            
        if best_template:
            response_text = best_template["text"]
        else:
            self.conversation_history.append({"role": "user", "content": user_input})
            self.vector_store_local = _add_turn_to_store(self.conversation_history[-1], self.vector_store_local, self.embedder)
            retrieved = _retrieve_relevant(user_input, self.vector_store_local, k=TOP_K)
            prompt = _build_contexted_prompt(retrieved, emotion, user_input)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=200, temperature=0.2, top_p=0.8,
                    do_sample=True, pad_token_id=self.tokenizer.eos_token_id
                )
            
            prompt_length = len(inputs['input_ids'][0])
            generated_tokens = out[0][prompt_length:]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            self.conversation_history.append({"role": "assistant", "content": response_text})
            self.vector_store_local = _add_turn_to_store(self.conversation_history[-1], self.vector_store_local, self.embedder)

        return response_text

