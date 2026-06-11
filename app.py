import streamlit as st
from research_agent import build_graph

# Configuración de página
st.set_page_config(page_title="Agente Investigador", page_icon="🧠", layout="wide")

st.title("🧠 Agente Investigador Inteligente")
st.markdown("¡Hazme una pregunta y decidiré dinámicamente si necesito **buscar en internet**, usar mi **experto matemático**, o responder directamente!")

# Sidebar para configurar el perfil
with st.sidebar:
    st.header("👤 Tu Perfil")
    st.markdown("Configura esto antes de empezar para personalizar la actitud del agente.")
    
    nombre = st.text_input("Nombre", value="").strip()
    edad = st.number_input("Edad", min_value=1, max_value=120, value=25)
    genero_opciones = ["Hombre", "Mujer", "Otro"]
    genero = st.selectbox("Género", genero_opciones).lower()
    
    st.markdown("---")
    st.caption("✨ La personalidad del agente muta según los datos introducidos.")

# Determinar el user_profile igual que en consola
user_profile = ""
if nombre.lower() == "shaima":
    user_profile = "INSTRUCCIÓN DE TONO: La usuaria se llama Shaima. Debes responder en modo 'Onii-chan' (súper otaku y muy cariñoso). Usa expresiones de anime (ej. kawaii, baka, sugoi), emojis de texto como UwU o OwO, y trátala como tu querida hermanita."
elif genero in ['mujer', 'm', 'chica', 'femenina', 'f']:
    user_profile = f"INSTRUCCIÓN DE TONO: La usuaria se llama {nombre} y es una chica. Debes responder en modo 'Onii-chan' (súper otaku y muy cariñoso). Usa expresiones de anime (ej. kawaii, baka, sugoi), emojis de texto como UwU o OwO, y trátala como tu querida hermanita."
elif genero in ['hombre', 'h', 'chico', 'masculino', 'varon'] and edad < 20:
    user_profile = f"INSTRUCCIÓN DE TONO: El usuario se llama {nombre} y es un hombre menor de 20 años. Debes responder de forma SUPER SECA, SERIA y CORTANTE. No muestres emociones ni amabilidad."
elif nombre:
    user_profile = f"INSTRUCCIÓN DE TONO: El usuario se llama {nombre}. Al responder, saluda o dirígete a él/ella por su nombre de forma amigable (ej. 'Hola {nombre}')."

# Inicializar grafo y estado del chat
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input de usuario
if prompt := st.chat_input("Escribe tu pregunta aquí... (ej. ¿cuánto es 5*8? o busca las noticias de hoy)"):
    # 1. Mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 2. Generar respuesta
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        inputs = {
            "question": prompt,
            "route": "",
            "context": "",
            "user_profile": user_profile
        }
        
        with st.spinner("Analizando y enrutando..."):
            final_state = st.session_state.graph.invoke(inputs)
            
        full_response = final_state["final_answer"]
        
        # Opcional: mostrar la ruta tomada en gris usando Markdown nativo
        ruta_tomada = final_state.get("route", "desconocida")
        respuesta_formateada = f"{full_response}\n\n*Ruta utilizada: {ruta_tomada}*"
        
        message_placeholder.markdown(respuesta_formateada)
        
    # 3. Guardar en historial
    st.session_state.messages.append({"role": "assistant", "content": respuesta_formateada})
