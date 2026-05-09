import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os
import time

# 1. Configuración de Hardware
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Entrenando usando: {device.type.upper()}")

# 2. Preparación de Datos y "Data Augmentation"
# Transformamos las imágenes para que la IA aprenda mejor
transformaciones = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # Voltea las imágenes al azar
    transforms.RandomRotation(15),     # Rota ligeramente
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Cargar la carpeta del dataset
data_dir = 'dataset_clinico'
dataset = datasets.ImageFolder(data_dir, transform=transformaciones)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

clases = dataset.classes
num_clases = len(clases)
print(f"📦 Se encontraron {len(dataset)} imágenes en {num_clases} categorías clínicas.")
print(f"📋 Clases: {clases}")

# 3. Construcción del Modelo Clínico
print("\n🧠 Descargando cerebro base (MobileNetV2)...")
modelo = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Congelamos las capas iniciales para que no olvide cómo ver bordes y colores
for param in modelo.parameters():
    param.requires_grad = False

# Cambiamos la última capa para que clasifique nuestras categorías clínicas
modelo.classifier[1] = nn.Linear(modelo.last_channel, num_clases)
modelo = modelo.to(device)

# 4. Configuración del Entrenamiento
criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.classifier.parameters(), lr=0.001)
epocas = 10 # 10 vueltas al dataset son suficientes para este caso

# Asegurar que la carpeta del modelo exista
ruta_guardado = "src/model"
if not os.path.exists(ruta_guardado):
    os.makedirs(ruta_guardado)

print("\n🚀 Iniciando entrenamiento...")
tiempo_inicio = time.time()

for epoca in range(epocas):
    modelo.train()
    perdida_total = 0.0
    correctos = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizador.zero_grad()
        outputs = modelo(inputs)
        loss = criterio(outputs, labels)
        loss.backward()
        optimizador.step()
        
        perdida_total += loss.item()
        _, predicciones = torch.max(outputs, 1)
        total += labels.size(0)
        correctos += (predicciones == labels).sum().item()
        
    precision = 100 * correctos / total
    print(f"⏳ Época [{epoca+1}/{epocas}] | Pérdida: {perdida_total/len(dataloader):.4f} | Precisión: {precision:.2f}%")

tiempo_fin = time.time()
print(f"\n✅ Entrenamiento completado en {(tiempo_fin - tiempo_inicio)/60:.2f} minutos.")

# 5. Guardar el nuevo cerebro clínico
ruta_modelo_final = os.path.join(ruta_guardado, "dietvision_clinico.pth")
torch.save(modelo.state_dict(), ruta_modelo_final)
print(f"💾 Nuevo modelo guardado exitosamente en: {ruta_modelo_final}")