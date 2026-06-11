# 🧠 LangGraph Research Agent

Un agente investigador inteligente construido con **LangGraph** y **LangChain** que utiliza un flujo de trabajo condicional para responder preguntas. El agente es capaz de decidir de forma autónoma si necesita buscar información actualizada en internet o si puede responder basándose en su propio conocimiento.

## ✨ Características Principales

- **Personalidad Adaptativa y Saludos Dinámicos:** El agente asume el rol de un **Especialista en Pokémon** y personaliza sus respuestas según tu perfil. Si indicas tu nombre, te saludará de forma amistosa por tu nombre en cada respuesta. Además, ajusta drásticamente su tono: responde de forma súper seca y seria si eres un hombre menor de 20 años, o entra en modo "Onii-chan" otaku y súper cariñoso si indicas que eres mujer.
- **Enrutamiento Inteligente a 3 Vías (Router):** Analiza la pregunta del usuario y decide dinámicamente si:
  1. Requiere una búsqueda web (eventos recientes o datos específicos).
  2. Requiere al **Experto en Matemáticas** (para cálculos numéricos y problemas).
  3. Puede responder directamente con conocimiento general.
- **Búsqueda Web Avanzada:** Integración con **Tavily** para extraer contexto y fuentes fiables en tiempo real.
- **Flexibilidad de Modelos (Foundry/Azure):** Compatible tanto con la API estándar de OpenAI como con **Azure OpenAI**, configurándose automáticamente según las variables de entorno.
- **Observabilidad (Tracing):** Configurado de fábrica con **LangSmith** para poder depurar, monitorizar y trazar cada paso y decisión que toma el agente en el grafo.

## 🛠️ Requisitos Previos

- Python 3.8+
- Claves de API de:
  - OpenAI o Azure OpenAI
  - Tavily (para búsqueda web)
  - LangSmith (opcional pero recomendado para el tracing)

## 🚀 Instalación y Configuración

1. **Clona el repositorio** e instala las dependencias (se recomienda usar un entorno virtual):
   ```bash
   pip install -r requirements.txt
   ```

2. **Configura las variables de entorno**:
   Copia el archivo `.env.example` a `.env`:
   ```bash
   cp .env.example .env
   ```

3. **Edita el archivo `.env`** con tus credenciales. 
   
   *Si usas Azure OpenAI:*
   ```env
   AZURE_OPENAI_API_KEY="tu_clave_api"
   AZURE_OPENAI_ENDPOINT="https://tu-recurso.cognitiveservices.azure.com/"
   OPENAI_API_VERSION="2025-01-01-preview"
   MODEL_NAME="gpt-4o-mini"
   ```

   *Si usas OpenAI estándar:*
   ```env
   OPENAI_API_KEY="tu_clave_openai"
   MODEL_NAME="gpt-4o-mini"
   ```

   *Otras configuraciones necesarias:*
   ```env
   TAVILY_API_KEY="tu_clave_de_tavily"
   
   # LangSmith
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
   LANGCHAIN_API_KEY="tu_clave_de_langsmith"
   LANGCHAIN_PROJECT="research_agent"
   ```

## 💻 Uso (Interfaz Web Premium)

Para arrancar la interfaz web interactiva con estética oscura y perfil lateral, ejecuta:

```bash
streamlit run app.py
```

Se abrirá automáticamente una pestaña en tu navegador web. 
1. En la barra lateral, configura tu **nombre**, **edad** y **género** para personalizar la actitud del agente.
2. Escribe tu pregunta en la barra inferior (ej. buscar en internet, resolver una operación matemática, o charlar de Pokémon).
3. El agente analizará tu pregunta y te mostrará su respuesta dinámica.

## 🏗️ Estructura del Grafo (LangGraph)

El flujo de trabajo consiste en los siguientes nodos:
1. `decide_search`: Analiza la pregunta y devuelve la ruta a seguir (`search_web`, `math_expert`, o `generate_answer`).
2. **Conditional Edge**: Enruta la ejecución hacia el nodo correspondiente según la decisión.
3. `search_web`: Ejecuta la búsqueda en Tavily y pasa el contexto a `generate_answer`.
4. `generate_answer`: Redacta la respuesta final basándose en el contexto obtenido o en el conocimiento general.
5. `math_expert`: Rama paralela especializada exclusiva para resolver problemas matemáticos directamente.