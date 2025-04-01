# GenIA - Application de Résumé d'Articles Académiques

Une application Streamlit qui utilise OpenAI et RAG pour résumer des articles académiques et répondre à des questions spécifiques.

## Fonctionnalités

- 📝 Génération automatique de résumés structurés (Introduction, Méthodologie, Résultats, Conclusion)
- ❓ Réponses aux questions spécifiques sur le contenu des articles
- 📚 Support des documents PDF académiques
- 🔍 Accès aux extraits sources utilisés pour générer les réponses

## Prérequis

- Python 3.8+
- Clé API OpenAI

## Installation

1. Clonez le dépôt :
```bash
git clone https://github.com/HajerGam/projet_genia.git
cd projet_genia
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Configurer la clé API OpenAI
Créer un fichier `.env` et ajouter votre clé API OpenAI :
```ini
OPENAI_API_KEY=your_openai_api_key
```

### 4. Lancer l'application
```bash
streamlit run app.py
```

## 📄 Structure du projet
```
GenIA
├── app.py              # Script principal de l'application
├── requirements.txt     # Dépendances Python
├── .env                # Configuration des variables d'environnement
├── README.md           # Documentation du projet
└── documents/          # Dossier pour stocker les PDFs uploadés
```

## 🔧 Technologies utilisées
- 🐍 Python
- 🚀 Streamlit (interface)
- 🤖 OpenAI (LLM)
- 🔍 LangChain (RAG)
- 🗃️ ChromaDB (vecteurs)

## 📢 Contribuer
Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 📜 Licence
Ce projet est sous licence MIT.

## Auteur
- HAjER GAM 
- Raha ATRI
- Moubarak Yahya Moussa
- Eya SAIDI