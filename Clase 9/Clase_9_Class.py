# %% [markdown]
# ## 1. Configuración Inicial y Preparación de Datos

# %%
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix
from PIL import Image
import numpy as np

# Configuración del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# %% [markdown]
# ### 1.1. Descarga y Estructuración del Dataset
# Aquí se debe incluir el código para descargar el dataset (por ejemplo, desde Roboflow).
# La estructura esperada localmente es: dataset/{train, valid, test}/{clase_1, clase_2, ...}

# %%
# PLACEHOLDER: Código para descargar el dataset desde Roboflow
# Ejemplo de uso de la API de Roboflow:
# !pip install roboflow
# from roboflow import Roboflow
# rf = Roboflow(api_key="TU_API_KEY")
# project = rf.workspace("tu_workspace").project("tu_proyecto")
# dataset = project.version(1).download("folder")

# Definimos las rutas a las particiones del dataset
data_dir = "./dataset" # Reemplazar con la ruta real post-descarga
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "valid")
test_dir = os.path.join(data_dir, "test")

# %% [markdown]
# ### 1.2. Transformaciones y DataLoaders
# Implementamos transformaciones 'on-the-fly' (Data Augmentation) para mejorar la generalización en entrenamiento,
# así como el reescalado a la dimensión esperada por la red (224x224).

# %%
# Transformaciones para entrenamiento (con Data Augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformaciones para validación y prueba (solo reescalado y normalización)
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Definición de la clase de Dataset personalizada
class CustomImageDataset(Dataset):
    """
    Dataset personalizado para cargar imágenes desde una estructura de carpetas (clases).
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Obtenemos las clases basadas en los nombres de las subcarpetas
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        # Recorremos cada clase y guardamos las rutas de sus imágenes
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if os.path.isfile(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Instanciación de los Datasets y DataLoaders
try:
    train_dataset = CustomImageDataset(train_dir, transform=train_transforms)
    valid_dataset = CustomImageDataset(valid_dir, transform=val_test_transforms)
    test_dataset = CustomImageDataset(test_dir, transform=val_test_transforms)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"Dataset cargado correctamente. Clases: {class_names}")
except FileNotFoundError:
    print("Advertencia: Directorio de dataset no encontrado. Ejecute la celda de descarga (1.1) primero.")
    # Valores por defecto para permitir la ejecución secuencial del resto del script a modo de prueba
    num_classes = 2
    class_names = ['clase_A', 'clase_B']
    train_loader, valid_loader, test_loader = None, None, None

# %% [markdown]
# ## 2. Definición del Modelo y Función de Pérdida

# %% [markdown]
# ### 2.1. Arquitectura del Modelo
# Se utiliza una red ResNet preentrenada, ajustando la última capa lineal al número de clases de nuestro problema.

# %%
def build_model(num_classes):
    """
    Construye y adapta una arquitectura ResNet18 para clasificación de `num_classes` clases.
    """
    # Cargamos ResNet18 con pesos preentrenados en ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Reemplazamos la capa final para coincidir con nuestro número de clases
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

model = build_model(num_classes).to(device)

# %% [markdown]
# ### 2.2. Implementación de Focal Loss
# Focal Loss mitiga el problema de desbalance de clases reduciendo el peso relativo de los ejemplos bien clasificados.

# %%
class FocalLoss(nn.Module):
    """
    Función de pérdida Focal Loss para clasificación multiclase.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        # Calculamos la entropía cruzada base (sin reducción)
        ce_loss = self.ce_loss(inputs, targets)
        
        # Probabilidad de la clase verdadera
        pt = torch.exp(-ce_loss)
        
        # Fórmula de Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Inicializamos la función de costo y el optimizador
criterion = FocalLoss(gamma=2.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %% [markdown]
# ## 3. Entrenamiento y Evaluación Continua

# %% [markdown]
# ### 3.1. Funciones de Métricas
# Función auxiliar para reportar el Macro Average Accuracy y el Accuracy por clase.

# %%
def compute_metrics(y_true, y_pred, classes):
    """
    Calcula el Accuracy por clase y el Macro Average Accuracy usando la matriz de confusión.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Previene divisiones por cero en caso de clases sin muestras
    class_totals = np.maximum(cm.sum(axis=1), 1)
    per_class_accuracies = cm.diagonal() / class_totals
    
    report = {cls: acc for cls, acc in zip(classes, per_class_accuracies)}
    macro_avg_acc = np.mean(per_class_accuracies)
    
    return macro_avg_acc, report

# %% [markdown]
# ### 3.2. Ciclo de Entrenamiento
# Bucle principal para entrenar y validar el modelo durante múltiples épocas.

# %%
def train_and_validate(model, train_loader, valid_loader, criterion, optimizer, epochs=5):
    """
    Ejecuta el entrenamiento y la validación, reportando métricas por época.
    """
    if train_loader is None or valid_loader is None:
        print("DataLoader no disponible. Saltando entrenamiento.")
        return

    for epoch in range(epochs):
        # Fase de entrenamiento
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            
        epoch_train_loss = train_loss / len(train_loader.dataset)
        train_macro_acc, _ = compute_metrics(train_targets, train_preds, class_names)
        
        # Fase de validación
        model.eval()
        valid_loss = 0.0
        valid_preds, valid_targets = [], []
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                
                valid_preds.extend(preds.cpu().numpy())
                valid_targets.extend(labels.cpu().numpy())
                
        epoch_valid_loss = valid_loss / len(valid_loader.dataset)
        valid_macro_acc, val_report = compute_metrics(valid_targets, valid_preds, class_names)
        
        # Reporte por época
        print(f"Época [{epoch+1}/{epochs}]")
        print(f"  Entrenamiento -> Pérdida: {epoch_train_loss:.4f} | Macro Avg Acc: {train_macro_acc:.4f}")
        print(f"  Validación    -> Pérdida: {epoch_valid_loss:.4f} | Macro Avg Acc: {valid_macro_acc:.4f}")
        
        # Reporte detallado al final de la validación de la última época
        if epoch == epochs - 1:
            print("\n--- Reporte Final de Validación ---")
            print(f"Macro Avg Accuracy: {valid_macro_acc:.4f}")
            print("Accuracy por clase:")
            for cls_name, acc in val_report.items():
                print(f"  {cls_name}: {acc:.4f}")

# Para ejecutar el entrenamiento de verdad (requiere datos), descomentar:
# train_and_validate(model, train_loader, valid_loader, criterion, optimizer, epochs=5)

# %% [markdown]
# ## 4. Evaluación Final

# %% [markdown]
# ### 4.1. Evaluación en el Conjunto de Prueba
# Se extraen las métricas finales (Macro Avg Acc y detalle por clase) utilizando datos nunca antes vistos.

# %%
def evaluate_on_test(model, test_loader, classes):
    """
    Evalúa el modelo final sobre el conjunto de prueba independiente.
    """
    if test_loader is None:
        print("DataLoader de prueba no disponible. Saltando evaluación.")
        return
        
    model.eval()
    test_preds, test_targets = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            
    test_macro_acc, test_report = compute_metrics(test_targets, test_preds, classes)
    
    print("\n" + "="*40)
    print("RESULTADOS EN CONJUNTO DE PRUEBA")
    print("="*40)
    print(f"Macro Average Accuracy: {test_macro_acc:.4f}\n")
    print("Desglose de Accuracy por clase:")
    for cls_name, acc in test_report.items():
        print(f"  - {cls_name}: {acc:.4f}")
    print("="*40)

# Para realizar la evaluación de verdad (requiere modelo entrenado y datos), descomentar:
# evaluate_on_test(model, test_loader, class_names)
