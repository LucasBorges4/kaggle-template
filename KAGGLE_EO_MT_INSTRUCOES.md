# Instruções para Rodar EoMT no Kaggle

Guia rápido para configurar o ambiente e executar o repositório EoMT dentro de um notebook Kaggle.

---

## Passo 1 — Criar o Notebook

1. Acesse [kaggle.com/code](https://www.kaggle.com/code)
2. Clique em **New Notebook**
3. Em **Settings → Accelerator**, selecione **GPU** (ou TPU se disponível)
4. Em **Add-ons**, conecte o GitHub se precisar clonar repositório privado

---

## Passo 2 — Clonar o Repositório

Cole esta célula no topo do notebook:

```python
import os, subprocess, sys

REPO_URL = "https://github.com/tue-mps/eomt.git"
DEST_DIR = "/kaggle/working/eomt"

if not os.path.exists(DEST_DIR):
    subprocess.run(["git", "clone", REPO_URL, DEST_DIR], check=True)

os.chdir(DEST_DIR)
if DEST_DIR not in sys.path:
    sys.path.insert(0, DEST_DIR)

print("Repositório clonado em:", os.getcwd())
```

---

## Passo 3 — Instalar Dependências

```python
!pip install -r /kaggle/working/eomt/requirements.txt -q
!pip install -q pytorch-lightning torchmetrics timm wandb
```

---

## Passo 4 — Verificar GPU

```python
import torch
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## Passo 5 — Treinar (1 GPU)

```python
import subprocess

subprocess.run([
    "python3", "main.py", "fit",
    "-c", "configs/dinov2/coco/panoptic/eomt_large_640.yaml",
    "--trainer.devices", "1",
    "--trainer.accelerator", "gpu",
    "--data.batch_size", "2",
    "--data.path", "/kaggle/input/SEU_DATASET",
    "--trainer.default_root_dir", "/kaggle/working/eomt/outputs",
    "--trainer.max_epochs", "10",
], check=True)
```

---

## Passo 6 — Validar / Inferência

```python
subprocess.run([
    "python3", "main.py", "validate",
    "-c", "configs/dinov2/coco/panoptic/eomt_large_640.yaml",
    "--trainer.devices", "1",
    "--trainer.accelerator", "gpu",
    "--data.batch_size", "2",
    "--data.path", "/kaggle/input/SEU_DATASET",
    "--model.ckpt_path", "/kaggle/working/eomt/outputs/last.ckpt",
], check=True)
```

---

## Caminhos Importantes no Kaggle

| Destino | Caminho |
|---------|---------|
| Código EoMT | `/kaggle/working/eomt` |
| Dataset (upload) | `/kaggle/input/SEU_DATASET` |
| Outputs / checkpoints | `/kaggle/working/eomt/outputs` |
| Arquivos persistentes | `/kaggle/working/` |

---

## Arquivos de Contexto

- `EO_MT_CONTEXTO_IA.md` — Contexto completo do repositório para IA
- `KAGGLE_EO_MT_INSTRUCOES.md` — Este guia prático