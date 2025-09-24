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

# LangChain imports
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    st.warning("⚠️ python-dotenv no está instalado. Instálalo con: pip install python-dotenv")

def limpiar(valor):
    if valor and valor.startswith('"') and valor.endswith('"'):
        return valor[1:-1]
    return valor

# --- Configuración del Cliente y Modelos de LangChain ---
# Only set environment variables if they exist
github_token = limpiar(os.getenv("GITHUB_TOKEN"))
github_base_url = limpiar(os.getenv("GITHUB_BASE_URL", "https://models.inference.ai.azure.com"))

if github_token:
    os.environ["OPENAI_API_KEY"] = github_token
    os.environ["OPENAI_API_BASE"] = github_base_url
else:
    st.error("❌ GITHUB_TOKEN environment variable is not set. Please check your .env file.")
    st.info("💡 Make sure your .env file contains: GITHUB_TOKEN=your_token_here")
    st.stop()

st.set_page_config(page_title="Evaluacion 1", page_icon="📊", layout="wide")

def initialize_client():
    if not github_token:
        st.error("❌ GitHub token not available")
        return None
    
    client = OpenAI(
        base_url=github_base_url,
        api_key=github_token
    )
    return client

def initialize_embeddings():
    """Initialize LangChain embeddings model"""
    if not github_token:
        st.error("❌ GitHub token not available for embeddings")
        return None
    
    try:
        # Modelo de embeddings (compatible con la API de OpenAI)
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        return embeddings
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        return None

def get_embeddings_langchain(embeddings_model, texts):
    """Get embeddings using LangChain"""
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
    """Get query embedding using LangChain"""
    try:
        embedding = embeddings_model.embed_query(query)
        return np.array(embedding)
    except Exception as e:
        st.error(f"Error getting query embedding: {str(e)}")
        return None

def evaluate_faithfulness(client, query, context, response):
    if not client:
        return 5.0
        
    eval_prompt = f"""Evalúa si la respuesta es fiel al contexto proporcionado.

Consulta: {query}

Contexto:
{context}

Respuesta:
{response}

¿La respuesta está basada únicamente en la información del contexto? 
Responde con un número del 1-10 donde:
- 1-3: Respuesta contradice o no está basada en el contexto
- 4-6: Respuesta parcialmente basada en el contexto
- 7-10: Respuesta completamente fiel al contexto

Responde SOLO con el número:"""

    try:
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.1,
            max_tokens=10
        )
        return float(result.choices[0].message.content.strip())
    except:
        return 5.0

def evaluate_relevance(client, query, response):
    if not client:
        return 5.0
        
    eval_prompt = f"""Evalúa qué tan relevante es la respuesta para la consulta.

Consulta: {query}

Respuesta: {response}

¿Qué tan bien responde la respuesta a la consulta?
Responde con un número del 1-10 donde:
- 1-3: Respuesta no relacionada o irrelevante
- 4-6: Respuesta parcialmente relevante
- 7-10: Respuesta muy relevante y útil

Responde SOLO con el número:"""

    try:
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0.1,
            max_tokens=10
        )
        return float(result.choices[0].message.content.strip())
    except:
        return 5.0

def evaluate_context_precision(client, query, retrieved_docs):
    if not client or not retrieved_docs:
        return 0.0
    
    relevant_count = 0
    for doc in retrieved_docs:
        eval_prompt = f"""¿Este documento es relevante para responder la consulta?

Consulta: {query}

Documento: {doc['document'][:300]}...

Responde SOLO 'SI' o 'NO':"""
        
        try:
            result = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0.1,
                max_tokens=5
            )
            if result.choices[0].message.content.strip().upper() == 'SI':
                relevant_count += 1
        except:
            pass
    
    return relevant_count / len(retrieved_docs)

def hybrid_search_with_metrics(query, documents, embeddings, embeddings_model, client, top_k=5):
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

def generate_response_with_metrics(client, query, context_docs):
    if not client:
        return "Error: Cliente no disponible", 0.0
        
    start_time = time.time()
    
    context = "".join([f"Documento {i+1}: {doc['document']}" 
                      for i, doc in enumerate(context_docs)])
    
    # --- PROMPT ADAPTADO PARA PETCO ---
    prompt = f"""Eres 'PetAsesor', un asistente virtual de PetCo, diseñado para brindar información sobre productos y cuidado de mascotas. Tu objetivo es ser amable, claro y preciso. Responde las preguntas basándote **únicamente** en la información del contexto. Si la información no es suficiente, debes informar al usuario que contacte con un experto de la tienda o un veterinario. **No inventes información** ni des consejos médicos que no estén respaldados por los datos.

Contexto:
{context}

Pregunta del cliente: {query}

Respuesta:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=150
        )
        
        generation_time = time.time() - start_time
        response_text = response.choices[0].message.content
        
        return response_text, generation_time
    except Exception as e:
        return f"Error generating response: {str(e)}", time.time() - start_time

def create_evaluation_dataset():
    # Conjunto de datos de evaluación para el caso de PetCo
    return [
        {
            "query": "¿Qué alimento le puedo dar a mi perro con alergias?",
            "expected_context": "alimento hipoalergénico, PetFood Care",
            "ground_truth": "Recomendamos nuestra línea de alimentos hipoalergénicos 'PetFood Care' que contiene salmón y arroz integral. Estos ingredientes son ideales para perros con sensibilidades alimentarias."
        },
        {
            "query": "¿Cuáles son los síntomas de la otitis en gatos?",
            "expected_context": "síntomas otitis gatos",
            "ground_truth": "La otitis en gatos puede causar sacudidas de cabeza, rascado de orejas, olor desagradable y enrojecimiento del canal auditivo."
        },
        {
            "query": "¿Qué es la línea PetFood Care?",
            "expected_context": "línea de alimentos PetFood Care",
            "ground_truth": "La línea de alimentos 'PetFood Care' es una marca de PetCo formulada con ingredientes como salmón y arroz integral, diseñada para mascotas con sensibilidades alimentarias."
        }
    ]

def log_interaction(query, response, metrics, context_docs):
    if 'interaction_logs' not in st.session_state:
        st.session_state.interaction_logs = []
    
    log_entry = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'response': response,
        'metrics': metrics,
        'context_count': len(context_docs),
        'context_scores': [doc.get('combined_score', 0) for doc in context_docs]
    }
    
    st.session_state.interaction_logs.append(log_entry)

def export_langsmith_format(logs):
    langsmith_data = []
    for log in logs:
        langsmith_data.append({
            "run_id": log['id'],
            "timestamp": log['timestamp'],
            "inputs": {"query": log['query']},
            "outputs": {"response": log['response']},
            "metrics": log['metrics'],
            "metadata": {
                "context_count": log['context_count'],
                "context_scores": log['context_scores']
            }
        })
    return langsmith_data

def main():
    st.title("📊 RAG con Evaluación y Monitoreo (LangChain)")
    st.write("Sistema RAG con métricas detalladas usando LangChain para embeddings")
    
    if not github_token:
        st.error("❌ Please check your .env file and make sure GITHUB_TOKEN is set.")
        st.info("💡 Your .env file should contain: GITHUB_TOKEN=your_token_here")
        return
    
    if "eval_rag" not in st.session_state:
        st.session_state.eval_rag = {
            # --- BASE DE DATOS ADAPTADA PARA PETCO ---
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
            'enable_logging': True
        }
    
    if 'interaction_logs' not in st.session_state:
        st.session_state.interaction_logs = []
    
    client = initialize_client()
    if not client:
        st.error("❌ Failed to initialize OpenAI client")
        return
    
    if st.session_state.eval_rag['embeddings_model'] is None:
        try:
            st.session_state.eval_rag['embeddings_model'] = initialize_embeddings()
        except Exception as e:
            st.error(f"Error inicializando embeddings: {str(e)}")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Consulta", "📄 Documentos", "📊 Métricas", "🧪 Evaluación", "📈 Analytics"])
    
    with tab1:
        st.header("💬 Consulta con Monitoreo (LangChain)")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input("Haz tu pregunta:")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                top_k = st.slider("Docs a recuperar:", 1, 12, 3)
            with col_b:
                eval_enabled = st.checkbox("Evaluación automática", value=True)
            with col_c:
                st.session_state.eval_rag['enable_logging'] = st.checkbox("Logging", value=True)
        
        with col2:
            if st.button("🔄 Generar Embeddings (LangChain)"):
                if st.session_state.eval_rag['documents'] and st.session_state.eval_rag['embeddings_model']:
                    with st.spinner("Generando embeddings con LangChain..."):
                        embeddings = get_embeddings_langchain(
                            st.session_state.eval_rag['embeddings_model'],
                            st.session_state.eval_rag['documents']
                        )
                        if embeddings is not None:
                            st.session_state.eval_rag['embeddings'] = embeddings
                            st.success("✅ Embeddings listos con LangChain")
                        else:
                            st.error("❌ Error generando embeddings")
                else:
                    st.warning("Modelo de embeddings no disponible")
        
        if st.button("🚀 Consultar con Métricas") and query:
            if st.session_state.eval_rag['embeddings'] is None:
                st.warning("Genera embeddings primero")
            elif st.session_state.eval_rag['embeddings_model'] is None:
                st.warning("Modelo de embeddings no inicializado")
            else:
                with st.spinner("Procesando con métricas..."):
                    results, retrieval_time = hybrid_search_with_metrics(
                        query, 
                        st.session_state.eval_rag['documents'],
                        st.session_state.eval_rag['embeddings'],
                        st.session_state.eval_rag['embeddings_model'],
                        client,
                        top_k
                    )
                    
                    if not results:
                        st.error("Error en la búsqueda")
                        return
                    
                    response, generation_time = generate_response_with_metrics(client, query, results)
                    
                    metrics = {
                        'retrieval_time': retrieval_time,
                        'generation_time': generation_time,
                        'total_time': retrieval_time + generation_time,
                        'docs_retrieved': len(results),
                        'avg_relevance_score': np.mean([r['combined_score'] for r in results])
                    }
                    
                    if eval_enabled:
                        context_text = "".join([r['document'] for r in results])
                        
                        with st.spinner("Evaluando calidad..."):
                            metrics['faithfulness'] = evaluate_faithfulness(client, query, context_text, response)
                            metrics['relevance'] = evaluate_relevance(client, query, response)
                            metrics['context_precision'] = evaluate_context_precision(client, query, results)
                    
                    st.subheader("📋 Documentos Recuperados")
                    for i, result in enumerate(results):
                        with st.expander(f"Doc {i+1} - Score: {result['combined_score']:.3f}"):
                            st.write(result['document'])
                    
                    st.subheader("🤖 Respuesta")
                    st.write(response)
                    
                    st.subheader("⏱️ Métricas de Rendimiento")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Tiempo total", f"{metrics['total_time']:.2f}s")
                    with col2:
                        st.metric("Recuperación", f"{metrics['retrieval_time']:.2f}s")
                    with col3:
                        st.metric("Generación", f"{metrics['generation_time']:.2f}s")
                    with col4:
                        st.metric("Docs recuperados", metrics['docs_retrieved'])
                    
                    if eval_enabled:
                        st.subheader("🎯 Métricas de Calidad")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Fidelidad", f"{metrics['faithfulness']:.1f}/10")
                        with col2:
                            st.metric("Relevancia", f"{metrics['relevance']:.1f}/10")
                        with col3:
                            st.metric("Precisión contexto", f"{metrics['context_precision']:.2f}")
                    
                    if st.session_state.eval_rag['enable_logging']:
                        log_interaction(query, response, metrics, results)
    
    with tab2:
        st.header("📄 Gestión de Documentos")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📚 Documentos Actuales")
            
            for i, doc in enumerate(st.session_state.eval_rag['documents']):
                with st.expander(f"Documento {i+1} ({len(doc)} caracteres)"):
                    st.text_area(
                        f"Contenido del documento {i+1}:",
                        value=doc,
                        height=100,
                        key=f"doc_display_{i}",
                        disabled=True
                    )
                    
                    col_edit, col_delete = st.columns(2)
                    with col_edit:
                        if st.button(f"✏️ Editar", key=f"edit_{i}"):
                            st.session_state[f'editing_doc_{i}'] = True
                    
                    with col_delete:
                        if st.button(f"🗑️ Eliminar", key=f"delete_{i}"):
                            st.session_state.eval_rag['documents'].pop(i)
                            st.session_state.eval_rag['embeddings'] = None
                            st.rerun()
                    
                    if st.session_state.get(f'editing_doc_{i}', False):
                        new_content = st.text_area(
                            "Editar contenido:",
                            value=doc,
                            height=150,
                            key=f"edit_content_{i}"
                        )
                        
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.button(f"💾 Guardar", key=f"save_{i}"):
                                st.session_state.eval_rag['documents'][i] = new_content
                                st.session_state[f'editing_doc_{i}'] = False
                                st.session_state.eval_rag['embeddings'] = None
                                st.success("Documento actualizado")
                                st.rerun()
                        
                        with col_cancel:
                            if st.button(f"❌ Cancelar", key=f"cancel_{i}"):
                                st.session_state[f'editing_doc_{i}'] = False
                                st.rerun()
        
        with col2:
            st.subheader("➕ Agregar Documento")
            
            new_doc = st.text_area(
                "Contenido del nuevo documento:",
                height=200,
                placeholder="Escribe aquí el contenido del nuevo documento..."
            )
            
            if st.button("📝 Agregar Documento"):
                if new_doc.strip():
                    st.session_state.eval_rag['documents'].append(new_doc.strip())
                    st.session_state.eval_rag['embeddings'] = None
                    st.success("Documento agregado exitosamente")
                    st.rerun()
                else:
                    st.warning("El documento no puede estar vacío")
            
            st.subheader("📊 Estadísticas")
            st.metric("Total documentos", len(st.session_state.eval_rag['documents']))
            
            if st.session_state.eval_rag['documents']:
                avg_length = np.mean([len(doc) for doc in st.session_state.eval_rag['documents']])
                st.metric("Longitud promedio", f"{avg_length:.0f} caracteres")
                
                total_words = sum([len(doc.split()) for doc in st.session_state.eval_rag['documents']])
                st.metric("Total palabras", f"{total_words:,}")
            
            st.subheader("🔄 Acciones")
            
            if st.button("🗑️ Limpiar Todos"):
                if st.session_state.eval_rag['documents']:
                    st.session_state.eval_rag['documents'] = []
                    st.session_state.eval_rag['embeddings'] = None
                    st.success("Todos los documentos eliminados")
                    st.rerun()
            
            uploaded_file = st.file_uploader(
                "📁 Cargar archivo de texto",
                type=['txt', 'md'],
                help="Sube un archivo .txt o .md para agregarlo como documento"
            )
            
            if uploaded_file is not None:
                try:
                    content = uploaded_file.read().decode('utf-8')
                    if st.button("📥 Importar Archivo"):
                        st.session_state.eval_rag['documents'].append(content)
                        st.session_state.eval_rag['embeddings'] = None
                        st.success(f"Archivo '{uploaded_file.name}' importado exitosamente")
                        st.rerun()
                except Exception as e:
                    st.error(f"Error al leer el archivo: {str(e)}")
            
            st.subheader("🔧 Estado")
            if st.session_state.eval_rag['embeddings'] is not None:
                st.success("✅ Embeddings generados")
            else:
                st.warning("⚠️ Embeddings no generados")
                st.info("Genera embeddings después de modificar documentos")
    
    with tab3:
        st.header("📊 Dashboard de Métricas")
        
        if st.session_state.interaction_logs:
            df = pd.DataFrame([
                {
                    'timestamp': log['timestamp'],
                    'query_length': len(log['query']),
                    'response_length': len(log['response']),
                    **log['metrics']
                }
                for log in st.session_state.interaction_logs
            ])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Response Time")
                fig = px.line(df, x='timestamp', y='total_time', title="Total Time per Query")
                st.plotly_chart(fig, use_container_width=True)
                
                if 'faithfulness' in df.columns:
                    st.subheader("Faithfulness Distribution")
                    fig = px.histogram(df, x='faithfulness', title="Faithfulness Scores Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Retrieval Metrics")
                fig = px.scatter(df, x='retrieval_time', y='generation_time', 
                                 size='docs_retrieved', title="Retrieval vs Generation Time")
                st.plotly_chart(fig, use_container_width=True)
                
                if 'relevance' in df.columns and 'context_precision' in df.columns:
                    st.subheader("Quality vs Precision")
                    fig = px.scatter(df, x='context_precision', y='relevance', 
                                     title="Context Precision vs Relevance")
                    st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📈 General Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Queries", len(df))
            with col2:
                st.metric("Average Time", f"{df['total_time'].mean():.2f}s")
            with col3:
                if 'faithfulness' in df.columns:
                    st.metric("Average Faithfulness", f"{df['faithfulness'].mean():.1f}/10")
            with col4:
                if 'relevance' in df.columns:
                    st.metric("Average Relevance", f"{df['relevance'].mean():.1f}/10")
        else:
            st.info("No hay datos de interacciones aún. Realiza algunas consultas primero.")
    
    with tab4:
        st.header("🧪 Evaluación Sistemática")
        
        if st.button("🧪 Ejecutar Evaluación Completa"):
            if st.session_state.eval_rag['embeddings'] is None:
                st.warning("Genera embeddings primero")
            elif st.session_state.eval_rag['embeddings_model'] is None:
                st.warning("Modelo de embeddings no inicializado")
            else:
                eval_dataset = create_evaluation_dataset()
                results = []
                
                with st.spinner("Ejecutando evaluación sistemática..."):
                    for test_case in eval_dataset:
                        query = test_case['query']
                        
                        docs, retrieval_time = hybrid_search_with_metrics(
                            query,
                            st.session_state.eval_rag['documents'],
                            st.session_state.eval_rag['embeddings'],
                            st.session_state.eval_rag['embeddings_model'],
                            client,
                            3
                        )
                        
                        if docs:
                            response, generation_time = generate_response_with_metrics(client, query, docs)
                            
                            context_text = "".join([d['document'] for d in docs])
                            faithfulness = evaluate_faithfulness(client, query, context_text, response)
                            relevance = evaluate_relevance(client, query, response)
                            context_precision = evaluate_context_precision(client, query, docs)
                            
                            results.append({
                                'query': query,
                                'response': response,
                                'retrieval_time': retrieval_time,
                                'generation_time': generation_time,
                                'faithfulness': faithfulness,
                                'relevance': relevance,
                                'context_precision': context_precision,
                                'ground_truth': test_case['ground_truth']
                            })
                
                if results:
                    st.subheader("📊 Resultados de Evaluación")
                    eval_df = pd.DataFrame(results)
                    st.dataframe(eval_df)
                    
                    st.subheader("📈 Métricas Promedio")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Fidelidad", f"{eval_df['faithfulness'].mean():.1f}/10")
                    with col2:
                        st.metric("Relevancia", f"{eval_df['relevance'].mean():.1f}/10")
                    with col3:
                        st.metric("Precisión", f"{eval_df['context_precision'].mean():.2f}")
                    with col4:
                        st.metric("Tiempo total", f"{(eval_df['retrieval_time'] + eval_df['generation_time']).mean():.2f}s")
                else:
                    st.error("No se pudieron obtener resultados de evaluación")
    
    with tab5:
        st.header("📈 Analytics y Exportación")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📤 Exportar Datos")
            
            if st.button("📊 Exportar para LangSmith"):
                if st.session_state.interaction_logs:
                    langsmith_data = export_langsmith_format(st.session_state.interaction_logs)
                    st.json(langsmith_data[:2])
                    
                    json_str = json.dumps(langsmith_data, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="💾 Descargar JSON LangSmith",
                        data=json_str,
                        file_name=f"langsmith_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                else:
                    st.info("No hay datos para exportar")
            
            if st.button("📊 Exportar CSV"):
                if st.session_state.interaction_logs:
                    df = pd.DataFrame([
                        {
                            'timestamp': log['timestamp'],
                            'query': log['query'],
                            'response': log['response'][:100] + "...",
                            **log['metrics']
                        }
                        for log in st.session_state.interaction_logs
                    ])
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="💾 Descargar CSV",
                        data=csv,
                        file_name=f"rag_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No hay datos para exportar")

if __name__ == "__main__":
    main()

