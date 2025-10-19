import os
from itertools import chain
from typing import List
import pdfplumber
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()


def load_documents() -> List[str]:
    """
    Load PDF documents from the data folder.
    Returns:
        List of document strings
    """
    #pdf_folder = "./data"
    pdf_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    print("pdf folder ",pdf_folder)
    pdf_files = [
        "env_prot_act_1986.pdf",
        "con_prot_act_2019.pdf",
        "it_act_2000.pdf"
    ]
    results = []

    #Reads all pages of each PDF and stores text in a list.
    #Each PDF corresponds to one document string.
    for pdf_file in pdf_files:
        path = os.path.join(pdf_folder, pdf_file)
        if not os.path.exists(path):
            print(f"PDF not found: {pdf_file}")
            continue

        doc_text = ""
      #  with fitz.Document(path) as pdf:
         #   for page in pdf:
       #         doc_text += page.get_text()
       # results.append(doc_text)

        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                doc_text += page.extract_text()
        results.append(doc_text)
        print(f"Loaded PDF: {pdf_file}, length: {len(doc_text)} characters")
    return results


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        if not self.llm:
            raise ValueError(
                "No valid API key found. Please set one of: "
                "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

        # Initialize vector database
        self.vector_db = VectorDB()

        # Create RAG prompt template
        # TODO: Implementing RAG prompt template
        #This template includes {context} from your PDFs and {question} from user input.

        self.prompt_template = ChatPromptTemplate.from_template(
                                """
                                Use the following context to answer the question as accurately as possible.
                                
                                Context:
                                {context}
                                
                                Question:
                                {question}
                                
                                Answer:
                                """
                                )

        # Create the chain
        self.chain = self.prompt_template | self.llm | StrOutputParser()

        print("RAG Assistant initialized successfully")

    @staticmethod
    def  _initialize_llm():
        """
        Initialize the LLM by checking for available API keys.
        Tries OpenAI, Groq, and Google Gemini in that order.
        """
        # Check for OpenAI API key
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def invoke(self, input: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant using GPT-4o-mini.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            Dictionary containing the answer and retrieved context
        """

        # TODO: Implement the RAG query pipeline
        # HINT: Use self.vector_db.search() to retrieve relevant context chunks
        # HINT: Combine the retrieved document chunks into a single context string
        # HINT: Use self.chain.invoke() with context and question to generate the response
        # HINT: Return a string answer from the LLM

        # 1️⃣ Retrieve relevant chunks
        search_results = self.vector_db.search(input, n_results=n_results)

        chunks = search_results.get("documents", [])
        flat_chunks = list(chain.from_iterable(chunks))
        if not chunks:
            return "No relevant information found in the documents."

        # 2️⃣ Combine chunks into single context
        context = "\n\n".join(flat_chunks)

        # 3️⃣ Use the prompt template
        prompt = self.prompt_template.format(context=context, question=input)

        # 4️⃣ Generate answer using the LLM
        llm_answer = self.llm.invoke(prompt)  # ChatOpenAI automatically handles the call

        return llm_answer.text


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        docs = load_documents()
        print(f"Loaded {len(docs)} sample documents")

        assistant.add_documents(docs)

        done = False

        while not done:
            question = input("Enter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                result = assistant.invoke(question)
                print(result)

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()
