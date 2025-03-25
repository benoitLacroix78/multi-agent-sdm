from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory

app = FastAPI()

@app.get("/simulate")
def simulate_project():
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    
    memory = ConversationBufferMemory()
    po = Tool(
        name="Product Owner",
        func=lambda x: f"[PO] Décrit les besoins métier : {x}",
        description="Spécifie les besoins"
    )
    dev = Tool(
        name="Développeur",
        func=lambda x: f"[Dev] Implémente une solution pour : {x}",
        description="Développe une fonctionnalité"
    )
    testeur = Tool(
        name="Testeur QA",
        func=lambda x: f"[QA] Vérifie et teste : {x}",
        description="Teste la feature"
    )

    agent = initialize_agent(
        [po, dev, testeur], llm, agent="zero-shot-react-description", verbose=True, memory=memory
    )

    result = agent.run("Créer une fonctionnalité de connexion utilisateur.")
    return {"result": result}
