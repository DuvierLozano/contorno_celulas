import os
import gc
import torch as t
from torch.utils.data import DataLoader
from torch import optim
from datasets import load_process
from model import get_model
from train import train
from predict import save_prediction
from plot_metrics import plot_results

# Configuración
DEVICE = 'cuda' if t.cuda.is_available() else 'cpu'
ENCODER = 'resnet101'
WEIGHTS = 'imagenet'
EPOCHS = 10
LR = 1e-3

img_path = 'train'
msk_path = 'train_mask'
pred_path = 'pred_masks'
os.makedirs(pred_path, exist_ok=True)

img_lst = sorted(os.listdir(img_path))

# Inicializar modelo y optimizador
model = get_model(ENCODER, WEIGHTS).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

loss_history = []
accuracy_history = []

# Entrenamiento por imagen
for idx, img_name in enumerate(img_lst):
    print(f"\nEntrenando con imagen: {img_name} ({idx + 1}/{len(img_lst)})")

    dataset = load_process(img_path, msk_path, ENCODER, idx)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for epoch in range(1, EPOCHS + 1):
        loss, acc = train(model, loader, optimizer, DEVICE)
        print(f"  Epoch {epoch}/{EPOCHS} — loss: {loss:.4f}, accuracy: {acc:.4f}")
        loss_history.append(loss)
        accuracy_history.append(acc)

    save_prediction(model, img_path, img_name, pred_path, ENCODER, DEVICE)

# Guardar modelo y gráficas
t.save(model.state_dict(), "U-Net_resnet101_FULL.pt")
plot_results(loss_history, accuracy_history)

# Limpieza final
del model, loader
gc.collect()
if t.cuda.is_available():
    t.cuda.empty_cache()
    t.cuda.reset_peak_memory_stats()
