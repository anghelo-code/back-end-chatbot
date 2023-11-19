from sentence_transformers import SentenceTransformer, util
import nltk
import random
from models.Model import Message

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

class ChatbotModel:
    def __init__(self):
        self.model = SentenceTransformer('dariolopez/roberta-base-bne-finetuned-msmarco-qa-es-mnrl-mn')
        with open('ai_algorithm/EPIIS_Corpus_Chatbot.txt', 'r', encoding='UTF-8', errors='ignore') as file:
            raw = file.read()
        self.corpus = nltk.sent_tokenize(raw)
        self.corpus_embeddings = self.model.encode(self.corpus)

    def get_response(self, user_input):
        query_embedding = self.model.encode(user_input)
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=1)[0]
        for hit in hits:
            return self.corpus[hit['corpus_id']]

    def check_for_greeting(self, sentence: str ):
        SALUDOS_INPUTS = ("hola", "buenas", "saludos", "qué tal", "hey", "buenos días")
        SALUDOS_OUTPUTS = ["Hola", "Hola, ¿Qué tal?", "Hola, ¿Cómo te puedo ayudar?", "Hola, encantado de hablar contigo"]

        for word in sentence.split():
            if word.lower() in SALUDOS_INPUTS:
                return random.choice(SALUDOS_OUTPUTS)
        return None
