from flask import Flask, request, jsonify
from custom_agents.chat_planner_agent import chat_planner_agent
from custom_agents.prompt_formatter import PromptFormatter
from transformers import AutoTokenizer
import os
from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from kerykeion import AstrologicalSubject, Report
import google.generativeai as genai
import sys
import io
from langchain.prompts import PromptTemplate

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
myfile = None

def get_llm():
    api_key = os.environ['GROQ_API_KEY']
    llm = ChatGroq(temperature=0, groq_api_key=api_key, model_name="llama-3.1-70b-versatile")
    return llm

def get_long_ctx_llm():
    print("#33")
    api_key = os.environ['GEMINI_API_KEY']
    print("#34",api_key)
    genai.configure(api_key=api_key)

    llm = genai.GenerativeModel('gemini-1.5-flash')
    #myfile = llm.upload_file( "./historias/demo1-ejemplo-fisioterapia.txt")
    print("#35",str(myfile))
    return llm

def get_chat_expert(_llm, _tokenizer, _planning_llm, _long_ctx_llm):
    print("#45")
    return chat_planner_agent(_llm, _tokenizer, _planning_llm, _long_ctx_llm, log_level='DEBUG', log_file='./agentlogs/chat_planner.txt', logging_enabled=True)



print("#01")
llm = get_llm()
print("#02")
long_ctx_llm = get_long_ctx_llm()
print("#03")
planning_llm = llm
print("#04")
chat_expert = get_chat_expert(llm, tokenizer, planning_llm, long_ctx_llm)

app = Flask(__name__)




@app.route('/api/verify', methods=['POST'])
def verify():
    global chat_expert
    global genai
    global myfile
    print("#1")
    data = request.get_json()
    user = data.get('user')
    question = data.get('question')
    token = data.get('token')
    history = data.get('history')  # Obtener el historial de mensajes
    legajo = data.get('legajo')  # Obtener el legajo médico seleccionado
    
    print(f"#####Legajo recibido: {legajo}")
    # Aquí puedes procesar el historial de mensajes como necesites
    #print(f"Historial de mensajes: {history}")
    print(f"#99 user question: {question}")
    expert_answer = chat_expert.ask_question({"messages": history,"legajo": legajo,"query":question},genai,myfile)
        
    #####
    #response = model.generate_content([myfile,"\n\n" ,"dada la historia clinica del paciente y tus conocimientos medicos,"])

    #expert_answer = response.text

    print("#answer from assistant ", expert_answer)

    return jsonify({"message": expert_answer})


@app.route('/api/loaddocument', methods=['POST'])
def loaddocument():
    global myfile
    global long_ctx_llm
    try:
        HISTORIAS_DIR = "./historias"
        # Obtén el cuerpo JSON de la solicitud
        data = request.get_json()
        print("Datos recibidos en el webhook:", data)  # Debug de los datos recibidos

        # Obtén el nombre del legajo
        legajo = data.get('legajo')
        if not legajo:
            print("Error: No se recibió el nombre del legajo ",str(legajo))
            return jsonify({"error": "Nombre del legajo es obligatorio"}), 400

        # Ruta completa al archivo de la historia
        historia_path = os.path.join(HISTORIAS_DIR, f"{legajo}")

        # Verifica si el archivo existe
        if not os.path.isfile(historia_path):
            print("Error: Archivo no encontrado para el legajo:", legajo)
            return jsonify({"error": f"Archivo no encontrado para el legajo {legajo}"}), 404

        # Lee el contenido del archivo
        with open(historia_path, 'r', encoding='utf-8') as file:
            historia_content = file.read()

        print("Contenido del archivo de historia cargado:", historia_content)

        # Cargar el contenido en la caché de Google Gemini
        cache_key = f"./historias/{legajo}"
        print("#77 ", str(cache_key))
        myfile = genai.upload_file(cache_key)

        
        print("Contenido de la historia cacheado en Google Gemini con clave:", cache_key)

        return jsonify({"success": True, "message": "Documento cargado y cacheado correctamente en Google Gemini"})
    
    except Exception as e:
        print('Error al cargar el documento:', e)
        return jsonify({"error": "Error al cargar el documento"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
