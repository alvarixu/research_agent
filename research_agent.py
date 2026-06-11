import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# Cargar variables de entorno (claves API para OpenAI, Tavily, LangSmith)
load_dotenv()

# Definir el estado (State) de nuestro grafo
class AgentState(TypedDict):
    question: str
    search_needed: bool
    context: str
    final_answer: str

# Definir un modelo Pydantic para la salida estructurada de la decisión
class SearchDecision(BaseModel):
    search_needed: bool = Field(description="True if web search is needed to answer the question, False otherwise.")

def get_llm():
    """Inicializa el LLM correcto dependiendo si es Azure OpenAI o estándar."""
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return AzureChatOpenAI(
            azure_deployment=model_name,
            temperature=0
        )
    else:
        return ChatOpenAI(model=model_name, temperature=0)

def decide_search(state: AgentState):
    """Nodo que decide si se necesita una búsqueda en la web según la pregunta."""
    print("---DECIDIENDO SI SE NECESITA BÚSQUEDA---")
    question = state["question"]
    
    # Inicializar LLM
    llm = get_llm()
    
    # Usar salida estructurada para obtener una decisión booleana
    structured_llm = llm.with_structured_output(SearchDecision)
    
    system_prompt = """Eres un experto en enrutamiento (router). 
    Analiza la pregunta del usuario y decide si necesitas buscar en la web para responderla con precisión.
    Las preguntas sobre eventos recientes, hechos actuales o conocimientos muy específicos generalmente requieren búsqueda en la web.
    Las preguntas de conocimiento general, sintaxis de programación o lógica podrían no requerir búsqueda en la web."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    decision = structured_llm.invoke(messages)
    
    return {"search_needed": decision.search_needed}

def search_web(state: AgentState):
    """Nodo que realiza la búsqueda web usando Tavily."""
    print("---BUSCANDO EN LA WEB---")
    question = state["question"]
    
    # Inicializar la herramienta de búsqueda de Tavily
    tavily_tool = TavilySearchResults(max_results=3)
    
    # Realizar la búsqueda
    docs = tavily_tool.invoke({"query": question})
    
    # Formatear el contexto a partir de los resultados de búsqueda
    context = "\n\n".join([f"Fuente: {doc['url']}\nContenido: {doc['content']}" for doc in docs])
    
    return {"context": context}

def generate_answer(state: AgentState):
    """Nodo que redacta la respuesta final."""
    print("---GENERANDO RESPUESTA FINAL---")
    question = state["question"]
    context = state.get("context", "")
    
    llm = get_llm()
    
    if context:
        system_prompt = """Eres un asistente de investigación útil. 
        Responde a la pregunta del usuario utilizando el contexto proporcionado de una búsqueda en la web.
        Si la respuesta no está en el contexto, simplemente di que no lo sabes según los resultados de búsqueda.
        Cita siempre las fuentes si las utilizas en tu respuesta."""
        prompt = f"Contexto:\n{context}\n\nPregunta: {question}"
    else:
        system_prompt = "Eres un asistente útil. Responde a la pregunta del usuario basándote en tu conocimiento general."
        prompt = f"Pregunta: {question}"
        
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    return {"final_answer": response.content}

def router(state: AgentState):
    """Enrutador de borde condicional (Conditional edge)."""
    if state["search_needed"]:
        print("-> Enrutando a: Búsqueda Web")
        return "search_web"
    else:
        print("-> Enrutando a: Generar Respuesta (Directo)")
        return "generate_answer"

def build_graph():
    """Construye y compila el LangGraph."""
    workflow = StateGraph(AgentState)

    # Añadir nodos
    workflow.add_node("decide_search", decide_search)
    workflow.add_node("search_web", search_web)
    workflow.add_node("generate_answer", generate_answer)

    # Añadir bordes (edges)
    workflow.set_entry_point("decide_search")
    
    workflow.add_conditional_edges(
        "decide_search",
        router,
        {
            "search_web": "search_web",
            "generate_answer": "generate_answer"
        }
    )
    
    workflow.add_edge("search_web", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # Compilar
    app = workflow.compile()
    return app

if __name__ == "__main__":
    app = build_graph()
    
    print("¡Bienvenido al Agente Investigador con LangGraph!")
    print("--------------------------------------------------")
    
    while True:
        user_query = input("\nIntroduce tu pregunta (o 'salir' para terminar): ")
        if user_query.lower() in ['salir', 'quit', 'exit', 'q']:
            break
            
        inputs = {"question": user_query, "search_needed": False, "context": ""}
        
        # Ejecutar el grafo
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"Nodo completado: '{key}'")
        
        # Obtener el estado final completo usando invoke
        final_state = app.invoke(inputs)
        print("\n=== RESPUESTA FINAL ===")
        print(final_state["final_answer"])
        print("=======================\n")
