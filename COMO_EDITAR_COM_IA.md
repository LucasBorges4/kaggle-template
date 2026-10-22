# Como fazer a IA editar os arquivos do repositório no Kaggle

Este guia mostra o passo a passo para que uma IA (ou você mesmo, copiando o que ela sugerir) consiga mexer nos arquivos de um repositório clonado dentro do ambiente Kaggle.

---

## 1. Clone o repositório alvo no Kaggle

No notebook do Kaggle, crie uma célula e rode:

```python
import os, subprocess, sys

# Exemplo: clonando o EoMT original
REPO_URL = "https://github.com/tue-mps/eomt.git"
DEST_DIR = "/kaggle/working/eomt"

if not os.path.exists(DEST_DIR):
    subprocess.run(["git", "clone", REPO_URL, DEST_DIR], check=True)

# Se for seu próprio fork/repositório, use sua URL:
# REPO_URL = "https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git"

os.chdir(DEST_DIR)
if DEST_DIR not in sys.path:
    sys.path.insert(0, DEST_DIR)

print("Diretório:", os.getcwd())
print("Arquivos:", os.listdir("."))
```

> **Dica:** se for editar e enviar de volta, clone seu **próprio fork** do GitHub, não o repositório original (você não tem permissão de push no `tue-mps/eomt`).

---

## 2. Forneça contexto da estrutura para a IA

Cole o conteúdo de `EO_MT_CONTEXTO_IA.md` (já enviado ao repositório `kaggle-template`) como contexto da IA antes de pedir edições. Isso faz com que ela saiba o que cada arquivo faz.

Arquivos principais do EoMT que a IA provavelmente vai editar:

| Arquivo / Pasta | O que é |
|-----------------|---------|
| `main.py` | Ponto de entrada (PyTorch Lightning: `fit`, `validate`) |
| `configs/*.yaml` | Hiperparâmetros e caminhos |
| `src/` | Código do modelo (backbone, head, losses) |
| `requirements.txt` | Dependências Python |

---

## 3. Como a IA edita os arquivos (via código Python)

No Kaggle, a IA não tem acesso direto ao teclado do notebook, mas você pode copiar o código que ela sugerir e colar em uma célula.

### Exemplo A — Ler um arquivo

```python
caminho = "/kaggle/working/eomt/main.py"
with open(caminho, "r", encoding="utf-8") as f:
    conteudo = f.read()
print(conteudo[:2000])
```

### Exemplo B — Escrever/substituir conteúdo de um arquivo

```python
caminho = "/kaggle/working/eomt/configs/dinov2/coco/panoptic/eomt_large_640.yaml"

novo_conteudo = """
seed: 42
device: cuda
epochs: 5
batch_size: 2
learning_rate: 2e-5
output_dir: /kaggle/working/eomt/outputs
"""

with open(caminho, "w", encoding="utf-8") as f:
    f.write(novo_conteudo)

print(f"Arquivo atualizado: {caminho}")
```

### Exemplo C — Editar trecho específico com Python (substituição de texto)

```python
caminho = "/kaggle/working/eomt/train.py"

with open(caminho, "r", encoding="utf-8") as f:
    texto = f.read()

# Substituir um trecho
texto = texto.replace(
    "batch_size: 4",
    "batch_size: 2  # ajustado para 1 GPU do Kaggle"
)

with open(caminho, "w", encoding="utf-8") as f:
    f.write(texto)

print("Edição aplicada.")
```

### Exemplo D — Usar shell (`sed`, `cat`) dentro do notebook

```python
# Substituir texto via sed no terminal do Kaggle
!sed -i 's/batch_size: 4/batch_size: 2/g' /kaggle/working/eomt/configs/dinov2/coco/panoptic/eomt_large_640.yaml
```

---

## 4. Exemplo de prompt para enviar à IA

Copie e cole algo assim na conversa com a IA (fora do notebook, ou como célula de markdown para registro):

```
Você é um especialista em PyTorch Lightning e segmentação de imagem.
O repositório clonado está em /kaggle/working/eomt.
Estrutura principal: main.py (treino/validação), configs/*.yaml (hiperparâmetros), src/ (modelo).

Tarefa: modifique o arquivo configs/dinov2/coco/panoptic/eomt_large_640.yaml para funcionar com 1 GPU no Kaggle, reduzindo batch_size para 2 e alterando output_dir para /kaggle/working/eomt/outputs.

Retorne apenas o código Python que edita o arquivo, usando open() e write(). Não explique, só o código.
```

---

## 5. Commit e envio de volta ao GitHub (se for seu repositório)

Se você clonou um fork seu e quer salvar as alterações:

```python
import subprocess

# Configurar identidade do git (obrigatório para commit)
subprocess.run(["git", "config", "user.email", "seu@email.com"], check=True)
subprocess.run(["git", "config", "user.name", "Seu Nome"], check=True)

# Adicionar e commitar
subprocess.run(["git", "add", "."], check=True)
subprocess.run(["git", "commit", "-m", "Ajustes para rodar no Kaggle (1 GPU)"], check=True)

# Enviar para o GitHub (precisa de token com permissão de escrita)
# Use https://TOKEN@github.com/USUARIO/REPO.git ou configure SSH
subprocess.run(["git", "push", "origin", "main"], check=True)
```

> **Atenção:** não coloque seu token de acesso pessoal direto no notebook. Use as **Secrets do Kaggle** (Settings → Secrets) e acesse via `os.getenv("GITHUB_TOKEN")`.

---

## 6. Resumo do fluxo no Kaggle

1. Clone o repositório para `/kaggle/working/eomt`.
2. Cole o contexto (`EO_MT_CONTEXTO_IA.md`) para a IA entender o projeto.
3. Peça à IA para sugerir edições em arquivos específicos.
4. Copie o código Python sugerido para uma célula do notebook e execute.
5. Verifique se os arquivos mudaram (`!cat caminho/do/arquivo`).
6. (Opcional) Commit e push para seu próprio repositório no GitHub.

---

**Repositório com estes arquivos:** https://github.com/LucasBorges4/kaggle-template.git