import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate

# Charger les variables d'environnement
load_dotenv()

# Configuration de l'application
st.set_page_config(
    page_title="GenIA - Résumé d'Articles Académiques",
    page_icon="📚",
    layout="wide"
)

# Initialisation de l'état de session
if "processed" not in st.session_state:
    st.session_state.processed = False
    st.session_state.summary = None
    st.session_state.qa = None
    st.session_state.show_question_input = False

def process_document(file_path):
    # Charger le document PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    # Diviser le texte en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)
    
    # Créer les embeddings et la base vectorielle
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Créer la chaine QA avec RAG
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    prompt_template = """Vous êtes un assistant spécialisé dans la synthèse d'articles académiques.
    Utilisez les extraits suivants de l'article pour créer un résumé clair et structuré.
    Organisez le résumé en sections: Introduction, Méthodologie, Résultats, Conclusion.
    Soyez précis et conservez les termes techniques importants.

    Contexte: {context}
    Question: {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    
    return qa

def generate_summary():
    query = "Fournissez un résumé complet de l'article en structurant en sections: Introduction, Méthodologie, Résultats, Conclusion."
    with st.spinner("Génération du résumé en cours..."):
        try:
            # Utilisation de invoke() au lieu de __call__
            result = st.session_state.qa.invoke({"query": query})
            st.session_state.summary = result["result"]
            
            st.subheader("Résumé généré")
            st.markdown(result["result"])
            
            with st.expander("Voir les extraits utilisés"):
                for doc in result["source_documents"]:
                    st.markdown(f"**Page {doc.metadata['page']}**")
                    st.text(doc.page_content[:500] + "...")
                    st.markdown("---")
        except Exception as e:
            st.error(f"Erreur lors de la génération: {e}")

def ask_question(question):
    with st.spinner("Recherche de réponse..."):
        try:
            # Utilisation de invoke() au lieu de __call__
            result = st.session_state.qa.invoke({"query": question})
            
            st.subheader("Réponse")
            st.markdown(result["result"])
            
            with st.expander("Voir les extraits utilisés"):
                for doc in result["source_documents"]:
                    st.markdown(f"**Page {doc.metadata['page']}**")
                    st.text(doc.page_content[:500] + "...")
                    st.markdown("---")
        except Exception as e:
            st.error(f"Erreur lors de la recherche: {e}")

def main():
    st.title("📚 GenIA - Résumé d'Articles Académiques")
    st.markdown("""
    Cette application utilise l'IA pour générer des résumés structurés d'articles académiques en combinant 
    les techniques RAG (Retrieval-Augmented Generation) et les modèles de langage d'OpenAI.
    """)
    
    with st.sidebar:
        st.header("Paramètres")
        api_key = st.text_input("Clé API OpenAI", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        st.markdown("---")
        st.info("Pour utiliser cette application, vous devez fournir une clé API OpenAI.")
        st.markdown("Vous pouvez obtenir une clé API [ici](https://platform.openai.com/signup).")
    
    uploaded_file = st.file_uploader(
        "Télécharger un article académique (PDF)", 
        type=["pdf"]
    )
    
    if uploaded_file is not None:
        if not st.session_state.processed:
            file_path = f"./documents/{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            with st.spinner("Analyse de l'article en cours..."):
                try:
                    st.session_state.qa = process_document(file_path)
                    st.session_state.processed = True
                    st.success("Document prêt pour analyse!")
                except Exception as e:
                    st.error(f"Erreur lors du traitement: {e}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📝 Résumé de l'article"):
                generate_summary()
        
        with col2:
            if st.button("❓ Poser une question"):
                st.session_state.show_question_input = True
        
        if st.session_state.show_question_input:
            user_question = st.text_input("Posez votre question sur l'article:")
            if user_question:
                ask_question(user_question)
    else:
        st.info("Veuillez télécharger un fichier PDF pour commencer.")

if __name__ == "__main__":
    main()        
