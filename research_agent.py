import os
from typing import TypedDict, Literal
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# Cargar variables de entorno (claves API para OpenAI, Tavily, LangSmith)
# Se utiliza dotenv para cargar las variables definidas en el archivo .env
# Usamos override=True para asegurar que Streamlit recargue las nuevas variables
load_dotenv(override=True)

# Definir el estado (State) de nuestro grafo
# Esto dicta la estructura de la memoria y datos que fluyen entre los nodos
class AgentState(TypedDict):
    question: str      # La pregunta original del usuario
    route: str         # La ruta decidida a tomar (matemáticas, búsqueda web, respuesta general)
    context: str       # Contexto adicional obtenido (ej. resultados de búsqueda web)
    final_answer: str  # La respuesta final que se generó para el usuario
    user_profile: str  # El perfil del usuario (instrucciones de personalidad y tono)

# Definir un modelo Pydantic para la salida estructurada de la decisión de enrutamiento
# Ayuda a obligar al LLM a devolver un JSON predecible con únicamente una opción válida
class RouteDecision(BaseModel):
    route: Literal["search_web", "math_expert", "generate_answer"] = Field(
        description="Selecciona 'math_expert' si es un problema matemático. 'search_web' si requiere buscar información actual en internet. 'generate_answer' para conocimiento general."
    )

def get_llm():
    """Inicializa el LLM correcto dependiendo si es Azure OpenAI o estándar.
    Retorna la instancia configurada de LangChain (ChatOpenAI o AzureChatOpenAI)."""
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    # Verificamos si existe la variable para Azure; de lo contrario usamos estándar
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return AzureChatOpenAI(
            azure_deployment=model_name,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("OPENAI_API_VERSION", "2024-02-15-preview"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0  # Temperatura 0 para respuestas deterministas y precisas
        )
    else:
        return ChatOpenAI(model=model_name, temperature=0)

def decide_search(state: AgentState):
    """Nodo que decide si se necesita búsqueda, matemáticas o respuesta directa.
    Este es el primer paso del grafo (entry point)."""
    print("---ENRUTANDO PREGUNTA---")
    question = state["question"]
    
    # Inicializar el modelo de lenguaje (LLM)
    llm = get_llm()
    
    # Usar salida estructurada:
    # with_structured_output fuerza al LLM a devolver un objeto de tipo RouteDecision
    structured_llm = llm.with_structured_output(RouteDecision)
    
    # Prompt de sistema para darle el rol de 'enrutador' (router) al modelo
    system_prompt = """Eres un experto en enrutamiento (router). 
    Analiza la pregunta del usuario y decide la ruta adecuada:
    - 'math_expert': Para preguntas que requieran realizar operaciones matemáticas, cálculos numéricos o resolver problemas de álgebra/aritmética.
    - 'search_web': Para preguntas sobre eventos recientes, hechos actuales o conocimientos muy específicos que requieran búsqueda en la web.
    - 'generate_answer': Para preguntas de conocimiento general, lógica, programación o cualquier cosa que el modelo pueda responder directamente."""
    
    # Creamos el historial de mensajes que se envía al LLM para la toma de decisión
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    # Invocamos al modelo, quien evalúa la pregunta y devuelve la ruta
    decision = structured_llm.invoke(messages)
    
    # Actualizamos el estado devolviendo únicamente la ruta decidida
    return {"route": decision.route}

def search_web(state: AgentState):
    """Nodo que realiza la búsqueda web usando la API de Tavily.
    Solo se ejecuta si la ruta decidida en el nodo de enrutamiento fue 'search_web'."""
    print("---BUSCANDO EN LA WEB---")
    question = state["question"]
    
    # Inicializar la herramienta de búsqueda de Tavily
    # Se limita a 3 resultados (max_results) para no exceder la ventana de contexto
    tavily_tool = TavilySearchResults(max_results=3)
    
    # Realizar la búsqueda usando la pregunta original
    docs = tavily_tool.invoke({"query": question})
    
    # Extraer y formatear el contenido de los resultados
    # Combina URL y fragmentos de texto en un solo gran bloque de contexto
    context = "\n\n".join([f"Fuente: {doc['url']}\nContenido: {doc['content']}" for doc in docs])
    
    # Actualizamos el estado guardando este nuevo contexto
    return {"context": context}

def generate_answer(state: AgentState):
    """Nodo que redacta la respuesta final.
    Se ejecuta de forma directa si la ruta fue 'generate_answer', o después del nodo 'search_web'."""
    print("---GENERANDO RESPUESTA FINAL---")
    question = state["question"]
    context = state.get("context", "")
    user_profile = state.get("user_profile", "")
    
    # Inicializamos el modelo (LLM)
    llm = get_llm()
    
    # Construimos el prompt base, inyectando el perfil/personalidad de usuario si existe
    base_prompt = "Eres un asistente de investigación experto y de gran utilidad."
    if user_profile:
        base_prompt += f"\n{user_profile}"
    
    # Si tenemos contexto (es decir, el agente pasó antes por el nodo de búsqueda web)
    if context:
        system_prompt = f"""{base_prompt}
        Responde a la pregunta del usuario utilizando el contexto proporcionado de una búsqueda en la web.
        Si la respuesta no está en el contexto, simplemente di que no lo sabes según los resultados de búsqueda.
        Cita siempre las fuentes si las utilizas en tu respuesta."""
        # Se envía tanto la pregunta como el contexto
        prompt = f"Contexto:\n{context}\n\nPregunta: {question}"
    else:
        # Si no hay contexto, el LLM responde solo basándose en sus pesos / conocimiento interno
        system_prompt = f"{base_prompt}\nResponde a la pregunta del usuario basándote en tu conocimiento general."
        prompt = f"Pregunta: {question}"
        
    # Preparamos los mensajes
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
    
    # Generamos la respuesta con el LLM
    response = llm.invoke(messages)
    
    # Guardamos la respuesta final en el estado global
    return {"final_answer": response.content}

def generate_math_answer(state: AgentState):
    """Nodo experto especializado únicamente en resolver problemas matemáticos.
    Se ejecuta si el enrutador decidió tomar la ruta 'math_expert'."""
    print("---GENERANDO RESPUESTA MATEMÁTICA---")
    question = state["question"]
    user_profile = state.get("user_profile", "")
    
    # Inicializamos el LLM
    llm = get_llm()
    
    # Prompt base enfocado estrictamente en la precisión analítica y matemática
    base_prompt = "Eres un experto en matemáticas de nivel avanzado. Tu objetivo es resolver la siguiente operación o problema de manera clara."
    # Inyectamos el perfil de usuario (tono)
    if user_profile:
        base_prompt += f"\n{user_profile}"
        
    system_prompt = f"{base_prompt}\nAsegúrate de que los cálculos sean precisos y muestra los pasos si es apropiado."
    
    # Empaquetamos en el formato esperado
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Problema matemático a resolver: {question}")
    ]
    
    # Generamos la respuesta
    response = llm.invoke(messages)
    
    # Guardamos el resultado en final_answer
    return {"final_answer": response.content}

def router(state: AgentState):
    """Enrutador de borde condicional (Conditional edge).
    A diferencia de los nodos que modifican el estado, esta función lee el estado actual
    para dictar a qué nodo debe saltar LangGraph a continuación."""
    route = state.get("route", "generate_answer")
    
    # Lógica de bifurcación
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
    """Construye, conecta y compila el flujo de trabajo (workflow) usando LangGraph."""
    # Inicializamos el grafo de estado basado en la estructura definida en AgentState
    workflow = StateGraph(AgentState)

    # 1. Añadimos todos los nodos que componen nuestro agente (sus funciones correspondientes)
    workflow.add_node("decide_search", decide_search)
    workflow.add_node("search_web", search_web)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("math_expert", generate_math_answer)

    # 2. Definimos el punto de entrada principal (el nodo inicial del grafo)
    workflow.set_entry_point("decide_search")
    
    # 3. Añadimos un borde condicional desde 'decide_search' hacia el destino dinámico
    # Se utiliza la función 'router' para saber hacia donde dirigirse
    workflow.add_conditional_edges(
        "decide_search",
        router,
        {
            "search_web": "search_web",             # Si 'router' retorna 'search_web', salta a ese nodo
            "math_expert": "math_expert",           # Si retorna 'math_expert', salta a ese nodo
            "generate_answer": "generate_answer"    # Si retorna 'generate_answer', salta a ese nodo
        }
    )
    
    # 4. Añadimos los bordes normales (rutas estáticas)
    # Después de buscar en la web, el flujo siempre debe pasar a generar una respuesta
    workflow.add_edge("search_web", "generate_answer")
    # Después de generar una respuesta matemática, terminamos el grafo (END)
    workflow.add_edge("math_expert", END)
    # Después de generar la respuesta general, también terminamos (END)
    workflow.add_edge("generate_answer", END)

    # 5. Compilamos el grafo en un 'ejecutable' invocable (app)
    app = workflow.compile()
    return app

if __name__ == "__main__":
    # Instanciamos el grafo compilado (solo si se ejecuta este archivo desde consola)
    app = build_graph()
    
    print("¡Bienvenido al Agente Investigador con LangGraph!")
    print("--------------------------------------------------")
    
    # Capturamos datos del usuario para el sistema de perfilamiento dinámico
    nombre = input("Antes de empezar, ¿cuál es tu nombre?: ").strip()
    edad_str = input("Por favor indica tu edad: ")
    genero = input("Indica tu género (hombre/mujer/otro): ").strip().lower()
    
    # Validamos la edad introducida, asignando 99 por defecto en caso de error
    try:
        edad = int(edad_str.strip())
    except ValueError:
        edad = 99
        
    # Construimos el perfil de usuario basándonos en condiciones prestablecidas
    user_profile = ""
    if genero in ['mujer', 'm', 'chica', 'femenina', 'f']:
        user_profile = f"INSTRUCCIÓN DE TONO: La usuaria se llama {nombre} y es una chica. Debes responder en modo 'Onii-chan' (súper otaku y muy cariñoso). Usa expresiones de anime (ej. kawaii, baka, sugoi), emojis de texto como UwU o OwO, y trátala como tu querida hermanita."
    elif genero in ['hombre', 'h', 'chico', 'masculino', 'varon'] and edad < 20:
        user_profile = f"INSTRUCCIÓN DE TONO: El usuario se llama {nombre} y es un hombre menor de 20 años. Debes responder de forma SUPER SECA, SERIA y CORTANTE. No muestres emociones ni amabilidad."
    elif nombre:
        user_profile = f"INSTRUCCIÓN DE TONO: El usuario se llama {nombre}. Al responder, saluda o dirígete a él/ella por su nombre de forma amigable (ej. 'Hola {nombre}')."
    
    # Bucle principal de chat interactivo por terminal
    while True:
        user_query = input("\nIntroduce tu pregunta (o 'salir' para terminar): ")
        
        # Condición de salida limpia del bucle
        if user_query.lower() in ['salir', 'quit', 'exit', 'q']:
            break
            
        # Preparar los inputs iniciales inyectándolos al estado (AgentState)
        inputs = {"question": user_query, "route": "", "context": "", "user_profile": user_profile}
        
        # Ejecutar el grafo nodo por nodo usando stream() para visualizar el progreso y cada paso tomado
        for output in app.stream(inputs):
            for key, value in output.items():
                print(f"Nodo completado: '{key}'")
        
        # Una vez termina el stream, invocamos para obtener el estado consolidado de forma sencilla
        # (Nota: En producción sería más óptimo extraer los resultados directamente del 'output' final del stream)
        final_state = app.invoke(inputs)
        print("\n=== RESPUESTA FINAL ===")
        print(final_state["final_answer"])
        print("=======================\n")

