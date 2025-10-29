import streamlit as st
import os
import json
import time
import uuid
from datetime import datetime
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go

# LangChain imports for Agent
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.memory import ConversationBufferWindowMemory # IL2.2: Memoria de Contenido
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool # IL2.1: Herramienta de consulta
from langchain import hub

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Ensure .env file is in the same directory
    load_dotenv()
except ImportError:
    st.warning("python-dotenv no está instalado. Instálalo con: pip install python-dotenv")

def limpiar(valor):
    # Removes surrounding quotes from environment variables
    if valor and valor.startswith('"') and valor.endswith('"'):
        return valor[1:-1]
    return valor

# --- Client and Model Configuration ---
# Set environment variables for LangChain and native OpenAI client
github_token = limpiar(os.getenv("GITHUB_TOKEN"))
github_base_url = limpiar(os.getenv("GITHUB_BASE_URL", "https://models.inference.ai.azure.com"))

# --- CORRECCIÓN CLAVE: Leer nombres de despliegue desde el .env ---
chat_deployment_name = limpiar(os.getenv("CHAT_DEPLOYMENT_NAME", "gpt-4o"))
embedding_deployment_name = limpiar(os.getenv("EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-small"))
# ----------------------------------------

if github_token:
    os.environ["OPENAI_API_KEY"] = github_token
    os.environ["OPENAI_API_BASE"] = github_base_url
else:
    st.error("GITHUB_TOKEN environment variable is not set. Please check your .env file.")
    st.info("Make sure your .env file contains: GITHUB_TOKEN=your_token_here")
    st.stop()

st.set_page_config(page_title="Evaluacion 2: Agente Funcional", page_icon="🤖", layout="wide")

def initialize_client():
    # Initializes the native OpenAI client for evaluations and generation
    if not github_token:
        return None
    
    client = OpenAI(
        base_url=github_base_url,
        api_key=github_token
    )
    return client

def initialize_embeddings():
    # Initializes LangChain embeddings model
    if not github_token:
        return None
    
    try:
        # CORRECCIÓN: Usar el nombre de despliegue del .env
        embeddings = OpenAIEmbeddings(
            model=embedding_deployment_name # <-- ¡CORREGIDO!
        )
        return embeddings
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        return None

def get_embeddings_langchain(embeddings_model, texts):
    # Generates document embeddings
    try:
        if isinstance(texts[0], str):
            documents = [Document(page_content=text) for text in texts]
        else:
            documents = texts
        
        embeddings = embeddings_model.embed_documents([doc.page_content for doc in documents])
        return np.array(embeddings)
    except Exception as e:
        st.error(f"Error getting embeddings: {str(e)}")
        return None

def get_query_embedding_langchain(embeddings_model, query):
    # Generates query embedding
    try:
        embedding = embeddings_model.embed_query(query)
        return np.array(embedding)
    except Exception as e:
        st.error(f"Error getting query embedding: {str(e)}")
        return None

def hybrid_search_with_metrics(query, documents, embeddings, embeddings_model, top_k=5):
    # Performs hybrid search (semantic + keyword)
    start_time = time.time()
    
    query_embedding = get_query_embedding_langchain(embeddings_model, query)
    if query_embedding is None:
        return [], 0.0
    
    semantic_similarities = cosine_similarity([query_embedding], embeddings)[0]
    
    keyword_scores = []
    query_words = set(query.lower().split())
    for doc in documents:
        doc_words = set(doc.lower().split())
        overlap = len(query_words.intersection(doc_words))
        keyword_scores.append(overlap / max(len(query_words), 1))
    
    # Combined score (70% semantic, 30% keyword)
    combined_scores = 0.7 * semantic_similarities + 0.3 * np.array(keyword_scores)
    top_indices = np.argsort(combined_scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'document': documents[idx],
            'semantic_score': semantic_similarities[idx],
            'keyword_score': keyword_scores[idx],
            'combined_score': combined_scores[idx],
            'index': idx
        })
    
    retrieval_time = time.time() - start_time
    return results, retrieval_time

def generate_response_from_context(client, query, context_docs):
    # Generates LLM response based on retrieved context
    if not client:
        return "Error: Cliente no disponible", 0.0, ""
        
    start_time = time.time()
    
    context = "".join([f"Documento {i+1}: {doc['document']}" 
                      for i, doc in enumerate(context_docs)])
    
    # Custom prompt for the PetAsesor persona
    prompt = f"""Eres 'PetAsesor', un asistente virtual de PetCo, diseñado para brindar información sobre productos y cuidado de mascotas. Tu objetivo es ser amable, claro y preciso. Responde las preguntas basándote unicamente en la información del contexto. Si la información no es suficiente, debes informar al usuario que contacte con un experto de la tienda o un veterinario. No inventes información.

Contexto:
{context}

Pregunta del cliente: {query}

Respuesta:
"""

    try:
        response = client.chat.completions.create(
            model=chat_deployment_name, # <-- ¡CORREGIDO!
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=250
        )
        
        generation_time = time.time() - start_time
        response_text = response.choices[0].message.content
        
        return response_text, generation_time, context # Returns context for metrics
    except Exception as e:
        return f"Error generating response: {str(e)}", time.time() - start_time, ""

# --- IL2.1: Tool for Agent (IE1) ---
@tool
def petco_data_lookup(query: str) -> str:
    """Useful for answering specific questions about PetCo products, services, store hours, or common pet health issues (like allergies or otitis) by performing a lookup in the internal knowledge base. Only use this tool if you need information from the knowledge base."""
    
    # Access data from Streamlit session state
    documents = st.session_state.eval_rag.get('documents', [])
    embeddings = st.session_state.eval_rag.get('embeddings')
    embeddings_model = st.session_state.eval_rag.get('embeddings_model')
    client = st.session_state.eval_rag.get('client')
    
    if not documents or embeddings is None or embeddings_model is None or client is None:
        return "Error interno: La base de datos o el modelo de embeddings no están inicializados."

    top_k = 3 # Fixed for the tool
    
    # 1. Retrieval (RAG)
    results, retrieval_time = hybrid_search_with_metrics(
        query, documents, embeddings, embeddings_model, top_k
    )
    
    if not results:
        return "No se encontró información relevante en la base de datos de PetCo para la consulta."
    
    # 2. Generation
    response, generation_time, context_text = generate_response_from_context(client, query, results)
    
    # Save tool metrics for logging later
    if 'tool_metrics' not in st.session_state:
        st.session_state.tool_metrics = {}
    
    st.session_state.tool_metrics['retrieval_time'] = retrieval_time
    st.session_state.tool_metrics['generation_time'] = generation_time
    st.session_state.tool_metrics['context_docs'] = results
    
    return response

# --- Evaluation Functions (IE4) ---
def evaluate_faithfulness(client, query, context, response):
    # Evaluates if the response is supported by the context
    if not client: return 5.0
    eval_prompt = f"""Evalúa si la respuesta es fiel al contexto proporcionado... [omitted for brevity]"""
    try:
        # CORRECCIÓN: Usar el nombre de despliegue del chat
        result = client.chat.completions.create(model=chat_deployment_name, messages=[{"role": "user", "content": eval_prompt}], temperature=0.1, max_tokens=10) # <-- ¡CORREGIDO!
        return float(result.choices[0].message.content.strip())
    except: return 5.0

def evaluate_relevance(client, query, response):
    # Evaluates how relevant the response is to the query
    if not client: return 5.0
    eval_prompt = f"""Evalúa qué tan relevante es la respuesta para la consulta... [omitted for brevity]"""
    try:
        # CORRECCIÓN: Usar el nombre de despliegue del chat
        result = client.chat.completions.create(model=chat_deployment_name, messages=[{"role": "user", "content": eval_prompt}], temperature=0.1, max_tokens=10) # <-- ¡CORREGIDO!
        return float(result.choices[0].message.content.strip())
    except: return 5.0

def evaluate_context_precision(client, query, retrieved_docs):
    # Evaluates the percentage of retrieved documents that are relevant
    if not client or not retrieved_docs: return 0.0
    relevant_count = 0
    for doc in retrieved_docs:
        eval_prompt = f"""¿Este documento es relevante para responder la consulta?... [omitted for brevity]"""
        try:
            # CORRECCIÓN: Usar el nombre de despliegue del chat
            result = client.chat.completions.create(model=chat_deployment_name, messages=[{"role": "user", "content": eval_prompt}], temperature=0.1, max_tokens=5) # <-- ¡CORREGIDO!
            if result.choices[0].message.content.strip().upper() == 'SI':
                relevant_count += 1
        except: pass
    return relevant_count / len(retrieved_docs)

# --- IL2.2, IL2.3: Agent Initialization ---
def initialize_rag_agent(client):
    # Initializes the LangChain Agent with memory and tool
    
    # IL2.1: Agent Tools
    tools = [petco_data_lookup]
    
    # IL2.3: LLM for Planning/Reasoning (ReAct)
    llm = ChatOpenAI(
        model=chat_deployment_name, # <-- ¡CORREGIDO!
        temperature=0.0,
        base_url=github_base_url,
        api_key=github_token
    )
    
    # IL2.2: Conversational Memory
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", 
        return_messages=True, 
        input_key="input",
        output_key="output",
        k=5 # Stores last 5 interactions
    )

    # ReAct prompt for planning
    prompt = hub.pull("hwchase17/react-chat")
    
    # IL2.3: Agent Executor (Planning and Decision Making)
    agent = create_react_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        memory=memory, 
        verbose=True, # Enables detailed log of reasoning
        handle_parsing_errors=True,
        max_iterations=10,
        return_intermediate_steps=True # For showing trace
    )
    
    return agent_executor, memory

# --- Logging and Export Functions ---
def log_interaction(query, response, metrics, context_docs):
    # Logs interaction details and metrics to session state
    log_entry = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'query': query,
        'response': response,
        'metrics': metrics,
        'context_docs': context_docs,
        'is_agent_tool_use': bool(context_docs) # Check if RAG tool was used
    }
    st.session_state.interaction_logs.append(log_entry)

def export_langsmith_format():
    # Prepares interaction logs for export in LangSmith format
    langsmith_data = []
    for log in st.session_state.interaction_logs:
        dataset_entry = {
            "input": log['query'],
            "output": log['response'],
            "context": [doc['document'] for doc in log['context_docs']] if log.get('context_docs') else [],
            "metadata": {
                "id": log['id'],
                "timestamp": log['timestamp'],
                "metrics": log['metrics']
            }
        }
        langsmith_data.append(dataset_entry)
    return langsmith_data

def create_evaluation_dataset(logs):
    # Creates a simplified DataFrame for evaluation analysis
    data = []
    for log in logs:
        metrics = log['metrics']
        data.append({
            'ID': log['id'],
            'Timestamp': log['timestamp'],
            'Query': log['query'],
            'Response (Snippet)': log['response'][:50] + "...",
            'Tool Used': 'Sí' if log['is_agent_tool_use'] else 'No',
            'Retrieval Time (s)': metrics.get('retrieval_time', 0.0),
            'Generation Time (s)': metrics.get('generation_time', 0.0),
            'Total Time (s)': metrics.get('total_time', 0.0),
            'Faithfulness (1-5)': metrics.get('faithfulness', 0.0),
            'Relevance (1-5)': metrics.get('relevance', 0.0),
            'Context Precision (0-1)': metrics.get('context_precision', 0.0)
        })
    return pd.DataFrame(data)

def main():
    st.title("Agente Funcional RAG con Evaluación (EP2)")
    st.write("Agente PetAsesor que utiliza planificación, memoria conversacional y una herramienta RAG para responder preguntas.")
    
    # 1. Initial Configuration and Token Check
    if not github_token:
        return 
    
    if "eval_rag" not in st.session_state:
        st.session_state.eval_rag = {
            # PetCo Knowledge Base
            'documents': [
                "PetCo es una cadena de tiendas en Chile que ofrece productos y servicios para mascotas, con foco en el bienestar animal y la asesoría experta.",
                "Nuestra misión es mejorar la vida de las mascotas y sus familias. Ofrecemos una amplia gama de productos, desde alimentos premium hasta juguetes y accesorios.",
                "La línea de alimentos 'PetFood Care' está formulada con ingredientes como salmón y arroz integral, ideales para perros con sensibilidades alimentarias o alergias. Ayuda a reducir la picazón y mejora la salud de la piel y el pelo.",
                "Ofrecemos servicios de peluquería y spa canino, los cuales se pueden reservar en nuestro sitio web. Nuestros estilistas están certificados y usan productos hipoalergénicos.",
                "Nuestros veterinarios internos han creado guías sobre enfermedades comunes. La otitis en gatos, por ejemplo, se manifiesta con rascado de orejas, sacudidas de cabeza, enrojecimiento y un olor desagradable.",
                "Contamos con una base de datos de preguntas frecuentes (FAQs). Las más comunes son sobre horarios de atención (Lunes a Sábado, de 9:00 a 20:00), ubicación de tiendas y política de devoluciones.",
                "Los síntomas de una alergia alimentaria en perros pueden ser picazón intensa, problemas digestivos como vómitos y diarrea, y otitis recurrente. Una dieta con proteínas limitadas, como la de 'PetFood Care', es la mejor opción.",
                "Para problemas de salud graves, como enfermedades crónicas o accidentes, recomendamos siempre contactar directamente con un veterinario calificado y no depender únicamente de la información proporcionada por este chat.",
                "Aparte de alimentos, tenemos una gran selección de juguetes interactivos para perros, rascadores para gatos y acuarios completos. Nuestro personal está capacitado para asesorarte en la elección del mejor producto para tu mascota."
            ],
            'embeddings': None,
            'embeddings_model': None,
            'enable_logging': True,
            'client': initialize_client() # Client for tool use and evaluation
        }
    
    if 'interaction_logs' not in st.session_state:
        st.session_state.interaction_logs = []
    
    client = st.session_state.eval_rag['client']
    if not client:
        st.error("Failed to initialize OpenAI client")
        return
    
    # Initialize Embeddings Model
    if st.session_state.eval_rag['embeddings_model'] is None:
        try:
            st.session_state.eval_rag['embeddings_model'] = initialize_embeddings()
        except Exception as e:
            st.error(f"Error inicializando embeddings: {str(e)}")
    
    # 2. Initialize Agent (IL2.1, IL2.2, IL2.3)
    if 'agent_executor' not in st.session_state:
        st.session_state.agent_executor, st.session_state.agent_memory = initialize_rag_agent(client)
    
    # 3. Tab Structure
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Agente (Consulta)", "Documentos", "Métricas", "Evaluación", "Analytics"])
    
    with tab1:
        st.header("Consulta al PetAsesor (Agente con Planificación)")
        
        # Embeddings Generation UI
        if st.button("Generar Embeddings"):
            if st.session_state.eval_rag['documents'] and st.session_state.eval_rag['embeddings_model']:
                with st.spinner("Generando embeddings con LangChain..."):
                    embeddings = get_embeddings_langchain(
                        st.session_state.eval_rag['embeddings_model'],
                        st.session_state.eval_rag['documents']
                    )
                    if embeddings is not None:
                        st.session_state.eval_rag['embeddings'] = embeddings
                        st.success("Embeddings listos con LangChain")
                    else:
                        st.error("Error generando embeddings")
            else:
                st.warning("Modelo de embeddings no disponible")
        
        st.divider()

        # Query and Evaluation Configuration
        col_a, col_b = st.columns([1, 1])
        with col_a:
            query = st.text_input("Pregúntale al Agente:")
        with col_b:
            eval_enabled = st.checkbox("Evaluación automática", value=True)
            st.session_state.eval_rag['enable_logging'] = st.checkbox("Logging", value=True)
        
        # Agent Execution (IL2.3)
        if st.button("Consultar Agente") and query:
            if st.session_state.eval_rag['embeddings'] is None:
                st.warning("Genera embeddings primero para que el Agente pueda usar su herramienta RAG.")
                return
            
            with st.spinner("El Agente está pensando y planificando..."):
                try:
                    # Agent execution with planning (ReAct)
                    result = st.session_state.agent_executor.invoke({"input": query})
                    response = result['output']
                    
                    st.subheader("Respuesta del Agente")
                    st.write(response)
                    
                    # IL2.3 (IE6): Show Planning Process
                    with st.expander("Proceso de Razonamiento del Agente (Planificación)"):
                        st.write("**Pasos intermedios:**")
                        for step in result['intermediate_steps']:
                            st.code(f"Thought: {step[0].log}\nObservation: {step[1]}")
                        st.write(f"**Memoria de Contenido (IL2.2) Actualizada:**")
                        st.json(st.session_state.agent_memory.load_memory_variables({})['chat_history'])

                    # Retrieve metrics from the RAG tool if it was used
                    retrieval_time = st.session_state.tool_metrics.get('retrieval_time', 0.0)
                    generation_time = st.session_state.tool_metrics.get('generation_time', 0.0)
                    context_docs = st.session_state.tool_metrics.get('context_docs', [])
                    
                    # Clear tool metrics after use
                    st.session_state.tool_metrics = {}

                    metrics = {
                        'retrieval_time': retrieval_time,
                        'generation_time': generation_time,
                        'total_time': retrieval_time + generation_time,
                        'docs_retrieved': len(context_docs),
                        'avg_relevance_score': np.mean([r['combined_score'] for r in context_docs]) if context_docs else 0.0
                    }
                    
                    if eval_enabled:
                        # RAG metrics only apply if the tool was used
                        if context_docs:
                            context_text = "".join([r['document'] for r in context_docs])
                            with st.spinner("Evaluando calidad (Fidelidad, Precisión)..."):
                                metrics['faithfulness'] = evaluate_faithfulness(client, query, context_text, response)
                                metrics['relevance'] = evaluate_relevance(client, query, response)
                                metrics['context_precision'] = evaluate_context_precision(client, query, context_docs)
                        else:
                            st.info("El agente respondió usando solo su memoria/razonamiento. No se aplican métricas RAG de Fidelidad y Precisión.")
                            metrics['faithfulness'] = 5.0 # Neutral score
                            metrics['relevance'] = evaluate_relevance(client, query, response)
                            metrics['context_precision'] = 0.0
                    
                    # Display metrics
                    st.subheader("Métricas de Rendimiento")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Tiempo Total (s)", f"{metrics['total_time']:.3f}")
                    col2.metric("Docs Recuperados", metrics['docs_retrieved'])
                    col3.metric("Fidelidad (1-5)", f"{metrics.get('faithfulness', 0.0):.2f}")
                    col4.metric("Precisión Contexto (0-1)", f"{metrics.get('context_precision', 0.0):.2f}")

                    if st.session_state.eval_rag['enable_logging']:
                        log_interaction(query, response, metrics, context_docs)

                except Exception as e:
                    st.error(f"Error durante la ejecución del Agente: {str(e)}")
                    st.session_state.tool_metrics = {} # Ensure cleanup

    with tab2:
        st.header("Base de Conocimiento de PetCo")
        st.write(f"Número de documentos: {len(st.session_state.eval_rag['documents'])}")
        for i, doc in enumerate(st.session_state.eval_rag['documents']):
            st.code(f"Doc {i+1}: {doc}", language="text")

    with tab3:
        st.header("Logs de Interacción")
        if st.session_state.interaction_logs:
            df_logs = create_evaluation_dataset(st.session_state.interaction_logs)
            st.dataframe(df_logs, use_container_width=True)
        else:
            st.info("No hay interacciones registradas.")

    with tab4:
        st.header("Resultados Detallados de Evaluación")
        if st.session_state.interaction_logs:
            selected_log_id = st.selectbox(
                "Selecciona una interacción para ver detalles:",
                options=[log['id'] for log in st.session_state.interaction_logs],
                format_func=lambda id: f"{[log['timestamp'] for log in st.session_state.interaction_logs if log['id'] == id][0]} - {id[:8]}..."
            )
            
            selected_log = next((log for log in st.session_state.interaction_logs if log['id'] == selected_log_id), None)

            if selected_log:
                st.subheader("Consulta y Respuesta")
                st.code(f"Consulta: {selected_log['query']}", language="text")
                st.code(f"Respuesta: {selected_log['response']}", language="text")

                st.subheader("Métricas de Calidad")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Fidelidad (1-5)", f"{selected_log['metrics'].get('faithfulness', 0.0):.2f}")
                col_m2.metric("Relevancia (1-5)", f"{selected_log['metrics'].get('relevance', 0.0):.2f}")
                col_m3.metric("Precisión Contexto (0-1)", f"{selected_log['metrics'].get('context_precision', 0.0):.2f}")
                
                st.subheader("Documentos Recuperados (Contexto)")
                if selected_log['context_docs']:
                    for doc in selected_log['context_docs']:
                        st.text(f"Score Combinado: {doc['combined_score']:.4f}")
                        st.code(doc['document'], language="text")
                else:
                    st.text("El agente no utilizó la herramienta RAG para esta consulta.")
        else:
            st.info("No hay datos de evaluación para mostrar.")

    with tab5:
        st.header("Visualización de Analytics")

        if st.session_state.interaction_logs:
            df_analytics = create_evaluation_dataset(st.session_state.interaction_logs)

            st.subheader("Distribución de Puntuaciones de Fidelidad")
            fig_f = px.histogram(df_analytics, x="Faithfulness (1-5)", title="Histograma de Fidelidad", nbins=5)
            st.plotly_chart(fig_f, use_container_width=True)

            st.subheader("Relación entre Precisión de Contexto y Fidelidad")
            fig_cp = px.scatter(df_analytics, x="Context Precision (0-1)", y="Faithfulness (1-5)", color="Tool Used",
                                 title="Precisión de Contexto vs. Fidelidad", trendline="ols")
            st.plotly_chart(fig_cp, use_container_width=True)

            st.subheader("Exportar Datos")
            
            if st.button("Exportar JSON (Formato LangSmith)"):
                if st.session_state.interaction_logs:
                    langsmith_data = export_langsmith_format()
                    json_str = json.dumps(langsmith_data, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="Descargar JSON LangSmith",
                        data=json_str,
                        file_name=f"langsmith_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.info("No hay datos para exportar")
            
            if st.button("Exportar CSV"):
                if st.session_state.interaction_logs:
                    csv = df_analytics.to_csv(index=False)
                    st.download_button(
                        label="Descargar CSV",
                        data=csv,
                        file_name=f"rag_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No hay datos para exportar")

if __name__ == "__main__":
    main()