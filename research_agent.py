import os
from typing import TypedDict, Literal
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
    route: str
    context: str
    final_answer: str
    user_profile: str

# Definir un modelo Pydantic para la salida estructurada de la decisión
class RouteDecision(BaseModel):
    route: Literal["search_web", "math_expert", "generate_answer"] = Field(
        description="Selecciona 'math_expert' si es un problema matemático. 'search_web' si requiere buscar información actual en internet. 'generate_answer' para conocimiento general."
    )

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
    """Nodo que decide si se necesita búsqueda, matemáticas o respuesta directa."""
    print("---ENRUTANDO PREGUNTA---")
    question = state["question"]
    
    # Inicializar LLM
    llm = get_llm()
    
    # Usar salida estructurada
    structured_llm = llm.with_structured_output(RouteDecision)
    
    system_prompt = """Eres un experto en enrutamiento (router). 
    Analiza la pregunta del usuario y decide la ruta adecuada:
    - 'math_expert': Para preguntas que requieran realizar operaciones matemáticas, cálculos numéricos o resolver problemas de álgebra/aritmética.
    - 'search_web': Para preguntas sobre eventos recientes, hechos actuales o conocimientos muy específicos que requieran búsqueda en la web.
    - 'generate_answer': Para preguntas de conocimiento general, lógica, programación o cualquier cosa que el modelo pueda responder directamente."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    decision = structured_llm.invoke(messages)
    
    return {"route": decision.route}

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
    user_profile = state.get("user_profile", "")
    
    llm = get_llm()
    
    base_prompt = "Eres un especialista en Pokémon y un asistente de investigación experto."
    if user_profile:
        base_prompt += f"\n{user_profile}"
    
    if context:
        system_prompt = f"""{base_prompt}
        Responde a la pregunta del usuario utilizando el contexto proporcionado de una búsqueda en la web.
        Si la respuesta no está en el contexto, simplemente di que no lo sabes según los resultados de búsqueda.
        Cita siempre las fuentes si las utilizas en tu respuesta."""
        prompt = f"Contexto:\n{context}\n\nPregunta: {question}"
    else:
        system_prompt = f"{base_prompt}\nResponde a la pregunta del usuario basándote en tu conocimiento general."
        prompt = f"Pregunta: {question}"
        
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    
    return {"final_answer": response.content}

def generate_math_answer(state: AgentState):
    """Nodo experto en resolver problemas matemáticos."""
    print("---GENERANDO RESPUESTA MATEMÁTICA---")
    question = state["question"]
    user_profile = state.get("user_profile", "")
    
    llm = get_llm()
    
    base_prompt = "Eres un experto en matemáticas de nivel avanzado. Tu objetivo es resolver la siguiente operación o problema de manera clara."
    if user_profile:
        base_prompt += f"\n{user_profile}"
        
    system_prompt = f"{base_prompt}\nAsegúrate de que los cálculos sean precisos y muestra los pasos si es apropiado."
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Problema matemático a resolver: {question}")
    ]
    
    response = llm.invoke(messages)
    
    return {"final_answer": response.content}

def router(state: AgentState):
    """Enrutador de borde condicional (Conditional edge)."""
    route = state.get("route", "generate_answer")
    if route == "search_web":
        print("-> Enrutando a: Búsqueda Web")
        return "search_web"
    elif route == "math_expert":
        print("-> Enrutando a: Experto en Matemáticas")
        return "math_expert"
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
    workflow.add_node("math_expert", generate_math_answer)

    # Añadir bordes (edges)
    workflow.set_entry_point("decide_search")
    
    workflow.add_conditional_edges(
        "decide_search",
        router,
        {
            "search_web": "search_web",
            "math_expert": "math_expert",
            "generate_answer": "generate_answer"
        }
    )
    
    workflow.add_edge("search_web", "generate_answer")
    workflow.add_edge("math_expert", END)
    workflow.add_edge("generate_answer", END)

    # Compilar
    app = workflow.compile()
    return app

if __name__ == "__main__":
    app = build_graph()
    
    print("¡Bienvenido al Agente Investigador con LangGraph!")
    print("--------------------------------------------------")
    
    nombre = input("Antes de empezar, ¿cuál es tu nombre?: ").strip()
    edad_str = input("Por favor indica tu edad: ")
    genero = input("Indica tu género (hombre/mujer/otro): ").strip().lower()
    
    try:
        edad = int(edad_str.strip())
    except ValueError:
        edad = 99
        
    user_profile = ""
    if genero in ['mujer', 'm', 'chica', 'femenina', 'f']:
        user_profile = f"INSTRUCCIÓN DE TONO: La usuaria se llama {nombre} y es una chica. Debes responder en modo 'Onii-chan' (súper otaku y muy cariñoso). Usa expresiones de anime (ej. kawaii, baka, sugoi), emojis de texto como UwU o OwO, y trátala como tu querida hermanita."
    elif genero in ['hombre', 'h', 'chico', 'masculino', 'varon'] and edad < 20:
        user_profile = f"INSTRUCCIÓN DE TONO: El usuario se llama {nombre} y es un hombre menor de 20 años. Debes responder de forma SUPER SECA, SERIA y CORTANTE. No muestres emociones ni amabilidad."
    elif nombre:
        user_profile = f"INSTRUCCIÓN DE TONO: El usuario se llama {nombre}. Al responder, saluda o dirígete a él/ella por su nombre de forma amigable (ej. 'Hola {nombre}')."
    
    while True:
        user_query = input("\nIntroduce tu pregunta (o 'salir' para terminar): ")
        if user_query.lower() in ['salir', 'quit', 'exit', 'q']:
            break
            
        inputs = {"question": user_query, "route": "", "context": "", "user_profile": user_profile}
        
        # Ejecutar el grafo
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"Nodo completado: '{key}'")
        
        # Obtener el estado final completo usando invoke
        final_state = app.invoke(inputs)
        print("\n=== RESPUESTA FINAL ===")
        print(final_state["final_answer"])
        print("=======================\n")
