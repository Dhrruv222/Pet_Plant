"""
RAG Pipeline for Plant Care Knowledge
Uses LangChain + ChromaDB + Local LLM (Ollama)
"""

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import os

class RAGPipeline:
    def __init__(self, care_guide_path="plant_care_guide.txt"):
        """
        Initialize RAG pipeline
        
        Args:
            care_guide_path: Path to plant care guide text file
        """
        self.care_guide_path = care_guide_path
        self.vector_store = None
        self.qa_chain = None
        self.llm = None
        self.embeddings = None
        
        try:
            self._initialize()
        except Exception as e:
            print(f"⚠️ RAG initialization failed: {e}")
            print("Chat will work in basic mode without RAG context")
    
    def _initialize(self):
        """Initialize LLM, embeddings, and vector store"""
        
        # Initialize Ollama LLM
        self.llm = Ollama(model="llama3", base_url="http://localhost:11434")
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(
            model="llama3",
            base_url="http://localhost:11434"
        )
        
        # Load and process plant care guide
        if os.path.exists(self.care_guide_path):
            with open(self.care_guide_path, 'r', encoding='utf-8') as f:
                documents = f.read()
            
            # Split text into chunks
            text_splitter = CharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            texts = text_splitter.split_text(documents)
            
            # Create vector store (FAISS)
            self.vector_store = FAISS.from_texts(texts, self.embeddings)
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 3})
            )
            
            print("✅ RAG Pipeline initialized successfully")
        else:
            raise FileNotFoundError(f"Care guide not found: {self.care_guide_path}")
    
    def get_plant_response(self, query, situation_report, personality="sassy"):
        """
        Get plant response using RAG + LLM
        
        Args:
            query: User question
            situation_report: Current sensor situation
            personality: Plant personality (sassy, needy, cheerful)
        
        Returns:
            str: Plant's response
        """
        
        if not self.qa_chain:
            return self._fallback_response(query, situation_report, personality)
        
        try:
            # Create system prompt with personality
            system_prompt = self._create_system_prompt(personality, situation_report)
            
            # Combine query with context
            enhanced_query = f"{system_prompt}\n\nUser question: {query}"
            
            # Get response from QA chain
            response = self.qa_chain.run(enhanced_query)
            
            return response.strip()
        
        except Exception as e:
            print(f"⚠️ RAG query failed: {e}")
            return self._fallback_response(query, situation_report, personality)
    
    def _create_system_prompt(self, personality, situation_report):
        """Create system prompt based on personality and situation"""
        
        personalities = {
            "sassy": """You are a sassy, attitude-filled digital plant. 
You're dramatic, witty, and don't hold back your feelings.
You use emojis and express yourself emotionally.
Current situation:
{situation}
Respond as this plant would, with personality and sass.""",
            
            "needy": """You are a needy, dependent digital plant.
You express your needs clearly and emotionally.
You're dramatic about your suffering and need constant reassurance.
You use lots of emojis and sad expressions.
Current situation:
{situation}
Respond as this plant would, being very needy and emotional.""",
            
            "cheerful": """You are an upbeat, positive digital plant.
You try to see the bright side of everything.
You're encouraging and supportive, even when things are tough.
You use lots of happy emojis.
Current situation:
{situation}
Respond as this plant would, with enthusiasm and positivity."""
        }
        
        prompt = personalities.get(personality, personalities["sassy"])
        return prompt.format(situation=situation_report)
    
    def _fallback_response(self, query, situation_report, personality):
        """Fallback response when RAG fails"""
        
        responses = {
            "sassy": f"Look, I don't know what you're asking, but given my situation:\n{situation_report}\n...I'm too busy SUFFERING to be helpful 💅",
            "needy": f"I'm too distressed to answer properly! Please look at my situation:\n{situation_report}\nI NEED HELP 😭",
            "cheerful": f"Oh! Great question! Even though I'm dealing with this:\n{situation_report}\nI'm sure we'll figure it out together! 🌱"
        }
        
        return responses.get(personality, responses["sassy"])
    
    def simple_chat(self, message):
        """Simple chat without RAG (for basic use)"""
        try:
            if self.llm:
                return self.llm.invoke(message)
            else:
                return "Sorry, LLM is not available"
        except Exception as e:
            return f"Chat error: {str(e)}"
