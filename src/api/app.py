from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os
import requests

app = FastAPI(title="DietVision Hospital Dispatch")

# --- CONFIGURACIÓN EDAMAM API ---
EDAMAM_APP_ID = "12e33274"
EDAMAM_APP_KEY = "255b8daf50d603f5e8fa4a749db550b0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Nuevas Clases Clínicas (En orden alfabético)
clases = [
    'arroz_blanco', 'avena', 'caldo_de_pollo', 'ensalada_mixta', 
    'gelatina', 'jugo_de_manzana', 'omelette_sencillo', 
    'pescado_al_vapor', 'pollo_a_la_plancha', 'pure_de_papas'
]

# Diccionario de traducción de alta precisión para Edamam
traduccion_edamam = {
    'arroz_blanco': 'plain white rice',
    'avena': 'plain oatmeal',
    'caldo_de_pollo': 'clear chicken broth',
    'ensalada_mixta': 'mixed green salad',
    'gelatina': 'jelly',
    'jugo_de_manzana': 'apple juice',
    'omelette_sencillo': 'plain omelette',
    'pescado_al_vapor': 'steamed white fish',
    'pollo_a_la_plancha': 'grilled chicken breast',
    'pure_de_papas': 'mashed potatoes'
}

# 2. Carga del Modelo Clínico (10 clases)
modelo = models.mobilenet_v2()
modelo.classifier[1] = nn.Linear(modelo.last_channel, 10) 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(BASE_DIR, "../model/dietvision_clinico.pth")

if os.path.exists(RUTA_MODELO):
    modelo.load_state_dict(torch.load(RUTA_MODELO, map_location=device))
modelo = modelo.to(device)
modelo.eval()

pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.get("/")
async def inicio():
    html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DietVision | Auditoría de Cocina</title>
        <style>
            :root { --dark: #0f172a; --primary: #3b82f6; --danger: #ef4444; --warning: #f59e0b; --success: #10b981; --bg: #e2e8f0; }
            body { font-family: 'Segoe UI', system-ui, sans-serif; background: var(--bg); display: flex; justify-content: center; padding: 20px; }
            .card { background: white; max-width: 600px; width: 100%; padding: 25px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.1); border-top: 6px solid var(--dark); }
            .header { text-align: center; margin-bottom: 20px; border-bottom: 2px solid #f1f5f9; padding-bottom: 15px; }
            .header h1 { margin: 0; color: var(--dark); font-size: 1.6rem; text-transform: uppercase; letter-spacing: 1px; }
            .upload-zone { border: 2px dashed #94a3b8; padding: 20px; text-align: center; border-radius: 12px; cursor: pointer; background: #f8fafc; margin-bottom: 15px; transition: 0.3s; }
            .upload-zone:hover { background: #e2e8f0; border-color: var(--primary); }
            .preview-img { width: 100%; height: 200px; object-fit: cover; border-radius: 8px; display: none; }
            .btn { background: var(--dark); color: white; border: none; padding: 15px; border-radius: 8px; font-weight: bold; width: 100%; cursor: pointer; font-size: 1rem; text-transform: uppercase; transition: 0.2s; }
            .btn:hover { background: #1e293b; }
            .btn:disabled { background: #94a3b8; cursor: not-allowed; }
            
            /* UI de Resultados */
            .results { display: none; margin-top: 20px; }
            .plate-name { font-size: 1.5rem; text-align: center; color: var(--primary); font-weight: 900; text-transform: uppercase; margin-bottom: 15px; }
            
            .alert-box { padding: 12px; border-radius: 8px; font-weight: bold; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }
            .alert-danger { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
            .alert-success { background: #d1fae5; color: #047857; border: 1px solid #6ee7b7; }
            
            .macros-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center; margin-bottom: 15px; }
            .macro-item { background: #f1f5f9; padding: 10px; border-radius: 8px; }
            .macro-item span { display: block; font-size: 0.75rem; color: #64748b; font-weight: bold; text-transform: uppercase; }
            .macro-item b { font-size: 1.1rem; color: var(--dark); }
            
            .clinical-info { background: #f0fdf4; border: 1px solid #bbf7d0; padding: 15px; border-radius: 8px; }
            .clinical-info h4 { margin: 0 0 10px 0; color: #166534; font-size: 0.9rem; text-transform: uppercase; }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="header">
                <h1>🏥 DietVision Despacho</h1>
                <p style="margin: 5px 0 0 0; color: #64748b; font-size: 0.9rem;">Auditoría de Nutrición Hospitalaria</p>
            </div>

            <div class="upload-zone" onclick="document.getElementById('fileInput').click()">
                <img id="preview" class="preview-img">
                <div id="textPlacer" style="font-weight: bold; color: #475569;">📸 Tocar para escanear Bandeja</div>
                <input type="file" id="fileInput" hidden accept="image/*" onchange="showPreview(event)">
            </div>

            <button class="btn" onclick="analizarBandeja()" id="btnMain">EVALUAR RESTRICCIONES</button>

            <div id="results" class="results">
                <div class="plate-name" id="res_plato">-</div>
                
                <div id="alertBox" class="alert-box alert-success">
                    <span id="alertIcon">✅</span>
                    <span id="alertText">Apto general. Sin restricciones severas detectadas.</span>
                </div>

                <div class="macros-grid">
                    <div class="macro-item"><span>Calorías</span><b id="res_cal">-</b></div>
                    <div class="macro-item"><span>Sodio</span><b id="res_na">-</b></div>
                    <div class="macro-item"><span>Azúcar</span><b id="res_sug">-</b></div>
                    <div class="macro-item"><span>Grasas</span><b id="res_fat">-</b></div>
                </div>

                <div class="clinical-info">
                    <h4>📋 Instrucciones de Servido</h4>
                    <p style="margin: 0; font-size: 0.9rem; color: #15803d; font-weight: 500;" id="res_porciones">-</p>
                </div>
            </div>
        </div>

        <script>
            function showPreview(e) {
                const reader = new FileReader();
                reader.onload = () => { 
                    document.getElementById('preview').src = reader.result;
                    document.getElementById('preview').style.display = 'block';
                    document.getElementById('textPlacer').style.display = 'none';
                };
                reader.readAsDataURL(e.target.files[0]);
            }

            async function analizarBandeja() {
                const file = document.getElementById('fileInput').files[0];
                if(!file) return alert("Sube la foto de la bandeja primero.");

                const btn = document.getElementById('btnMain');
                btn.disabled = true; btn.innerText = "⚙️ Procesando Reglas Médicas...";

                const formData = new FormData();
                formData.append('file', file);

                try {
                    const response = await fetch('/predict/', { method: 'POST', body: formData });
                    const data = await response.json();

                    // --- AUDITORÍA FRONTEND (F12) ---
                    console.log("%c📊 DATOS DE AUDITORÍA EDAMAM RECIBIDOS:", "color: #10b981; font-weight: bold; font-size: 14px;");
                    console.table(data);

                    // Poblar la UI
                    document.getElementById('res_plato').innerText = data.plato;
                    document.getElementById('res_cal').innerText = data.calorias + " kcal";
                    document.getElementById('res_na').innerText = data.sodio_mg + " mg";
                    document.getElementById('res_sug').innerText = data.azucar_g + " g";
                    document.getElementById('res_fat').innerText = data.grasas_g + " g";
                    document.getElementById('res_porciones').innerText = data.recomendacion;

                    const alertBox = document.getElementById('alertBox');
                    if (data.alertas.length > 0) {
                        alertBox.className = "alert-box alert-danger";
                        document.getElementById('alertIcon').innerText = "🚨";
                        document.getElementById('alertText').innerHTML = "<b>CONTRAINDICADO PARA:</b><br>" + data.alertas.join("<br>");
                    } else {
                        alertBox.className = "alert-box alert-success";
                        document.getElementById('alertIcon').innerText = "✅";
                        document.getElementById('alertText').innerText = "Apto general. Sin restricciones severas.";
                    }

                    document.getElementById('results').style.display = 'block';
                } catch(e) { 
                    console.error(e);
                    alert("Error conectando con el servidor"); 
                } finally { 
                    btn.disabled = false; btn.innerText = "EVALUAR NUEVA BANDEJA"; 
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict/")
async def predecir(file: UploadFile = File(...)):
    # 1. Inferencia IA (Clasificación de Imagen)
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert('RGB')
    tensor = pipeline(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = modelo(tensor)
        _, idx = torch.max(out, 1)
        clase_id = clases[idx.item()]
    
    nombre_formateado = clase_id.replace('_', ' ').title()
    query_edamam = traduccion_edamam.get(clase_id, "food")
    
    # 2. Extracción de Nutrientes API Edamam (Normalizado a 100g)
    url = f"https://api.edamam.com/api/nutrition-data?app_id={EDAMAM_APP_ID}&app_key={EDAMAM_APP_KEY}&ingr=100g {query_edamam}"
    
    try:
        r = requests.get(url).json()
        
        # --- AUDITORÍA BACKEND (Terminal VS Code) ---
        print(f"\n[{nombre_formateado.upper()}] Consultando a Edamam: '100g {query_edamam}'")
        print(f"📡 API URL: {url}")
        
        # Extracción robusta de métricas vitales
        cal = r.get('calories', 0)
        nutri = r.get('totalNutrients', {})
        
        if cal == 0 and 'ingredients' in r and len(r['ingredients']) > 0:
            parsed = r['ingredients'][0].get('parsed', [{}])[0]
            nutri = parsed.get('nutrients', {})
            cal = nutri.get('ENERC_KCAL', {}).get('quantity', 0)

        fat = float(nutri.get('FAT', {}).get('quantity', 0))
        sodio = float(nutri.get('NA', {}).get('quantity', 0)) # mg
        azucar = float(nutri.get('SUGAR', {}).get('quantity', 0)) # g

        # 3. MOTOR DE REGLAS CLÍNICAS (Filtros de exclusión)
        alertas_medicas = []
        
        if sodio > 300: # Límite de sodio para dieta hospitalaria estándar
            alertas_medicas.append("• Pacientes Hipertensos (Alta en Sodio >300mg)")
        if azucar > 10: # Límite de azúcares simples
            alertas_medicas.append("• Pacientes con Diabetes (Alta en Azúcar >10g)")
        if fat > 15: # Límite de grasas
            alertas_medicas.append("• Pacientes con Dislipidemia (Alta en Grasas >15g)")

        # 4. Asignación de Porciones
        recomendacion = "Porción estándar: 200g - 250g (Plato Principal)"
        if clase_id in ['caldo_de_pollo', 'gelatina', 'jugo_de_manzana']:
            recomendacion = "Dieta Líquida: Servir 150ml a 200ml. Verificar tolerancia."
        elif clase_id in ['pure_de_papas', 'pescado_al_vapor', 'arroz_blanco']:
            recomendacion = "Dieta Blanda: Porción sugerida 150g. Servir a temperatura tibia."
        elif clase_id == 'ensalada_mixta':
            recomendacion = "Acompañamiento: 100g. Asegurar correcta desinfección."

        return {
            "plato": nombre_formateado,
            "calorias": round(cal, 1),
            "grasas_g": round(fat, 1),
            "sodio_mg": round(sodio, 1),
            "azucar_g": round(azucar, 1),
            "alertas": alertas_medicas,
            "recomendacion": recomendacion
        }
        
    except Exception as e:
        print(f"❌ Error de API: {e}")
        return {
            "plato": nombre_formateado, "calorias": 0, "grasas_g": 0, 
            "sodio_mg": 0, "azucar_g": 0, "alertas": ["Error de conexión con Base de Datos"],
            "recomendacion": "Requiere auditoría manual."
        }