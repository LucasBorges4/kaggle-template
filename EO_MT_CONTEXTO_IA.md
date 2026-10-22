# Contexto do Repositório EoMT para IA (Kaggle)

Este arquivo foi criado para ser lido por uma IA como contexto antes de editar, executar ou otimizar o repositório **tue-mps/eomt** dentro do ambiente Kaggle.

---

## 1. O que é o EoMT

**EoMT (Encoder-only Mask Transformer)** é um modelo de segmentação de imagem apresentado na CVPR 2025 (Highlight). A ideia central é reutilizar um Vision Transformer (ViT) puro — sem adaptadores e sem decoders complexos — para codificar patches de imagem e queries de segmentação como tokens.

- **Paper:** [Your ViT is Secretly an Image Segmentation Model](https://arxiv.org/abs/2503.19108)
- **Autores:** Tommie Kerssies, Niccolò Cavagnero, Alexander Hermans, Narges Norouzi, Giuseppe Averta, Bastian Leibe, Gijs Dubbelman, Daan de Geus
- **Instituições:** Eindhoven University of Technology, Polytechnic of Turin, RWTH Aachen University
- **Licença:** Apache-2.0 (via dependências como Detectron2, Mask2Former, etc.)

---

## 2. Estrutura do Repositório

```
eomt/
├── main.py                 # Ponto de entrada (PyTorch Lightning: fit / validate)
├── requirements.txt        # Dependências Python
├── inference.ipynb         # Notebook de inferência rápida com pesos pré-treinados
├── configs/                # Arquivos YAML de configuração
│   ├── dinov2/
│   │   └── coco/
│   │       ├── panoptic/eomt_large_640.yaml
│   │       ├── instance/eomt_large_640.yaml
│   │       └── semantic/eomt_large_640.yaml
│   └── dinov3/
│       └── ...
├── model_zoo/              # Documentação e links de pesos pré-treinados
│   ├── dinov2.md
│   └── dinov3.md
├── src/                    # Código-fonte do modelo (backbone, head, losses, etc.)
├── data/                   # Scripts/utilitários de dataset
├── notebooks/              # Outros notebooks auxiliares
└── README.md               # Documentação oficial do projeto
```

---

## 3. Como o Treinamento Funciona (PyTorch Lightning)

O treinamento usa **PyTorch Lightning** via linha de comando:

```bash
python3 main.py fit \
  -c configs/dinov2/coco/panoptic/eomt_large_640.yaml \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset
```

### Argumentos importantes:
| Argumento | Descrição |
|-----------|-----------|
| `-c` ou `--config` | Caminho para o arquivo YAML de configuração |
| `--trainer.devices` | Número de GPUs (usar `1` no Kaggle) |
| `--trainer.accelerator` | `gpu` ou `cpu` |
| `--data.batch_size` | Batch size por GPU |
| `--data.path` | Diretório contendo os `.zip` do dataset |
| `--model.ckpt_path` | Caminho para checkpoint (fine-tuning / validação) |
| `--model.load_ckpt_class_head` | `False` para ignorar a cabeça de classificação ao mudar de dataset |

### Configurações disponíveis:
- `configs/dinov2/coco/panoptic/eomt_large_640.yaml` — Panoptic segmentation COCO
- `configs/dinov2/coco/instance/eomt_large_640.yaml` — Instance segmentation COCO
- `configs/dinov2/coco/semantic/eomt_large_640.yaml` — Semantic segmentation COCO
- `configs/dinov3/...` — Versões com backbone DINOv3

---

## 4. Datasets Suportados

O EoMT lê arquivos **`.zip` diretamente**, sem necessidade de descompactar:

| Dataset | Arquivos necessários |
|---------|---------------------|
| **COCO** | `train2017.zip`, `val2017.zip`, `annotations_trainval2017.zip`, `panoptic_annotations_trainval2017.zip` |
| **ADE20K** | `ADEChallengeData2016.zip`, `annotations_instance.zip` |
| **Cityscapes** | Arquivos baixados com login no site oficial |

No Kaggle, os datasets podem ser:
- Baixados diretamente via `wget` dentro do notebook
- Usados de `/kaggle/input/SEU_DATASET` se enviados como Kaggle Dataset

---

## 5. Adaptação para Kaggle (Ambiente com 1 GPU)

O README original pressupõe **4 GPUs H100** com batch 4 por GPU (total 16). No Kaggle, geralmente há **1 GPU** (T4/P100/A100 dependendo do plano). Adaptações necessárias:

```python
--trainer.devices 1
--trainer.accelerator gpu
--data.batch_size 2  # ou 4, conforme VRAM disponível
--data.path /kaggle/input/SEU_DATASET
--trainer.default_root_dir /kaggle/working/eomt/outputs
```

### Memória VRAM no Kaggle:
- T4: ~16 GB
- P100: ~16 GB
- A100: ~40-80 GB (planos superiores)

Se ocorrer `CUDA out of memory`, reduza `--data.batch_size` para `1` ou ative mixed precision.

---

## 6. Pesos Pré-treinados (Model Zoo)

O repositório disponibiliza pesos pré-treinados para DINOv2 e DINOv3.

- **Documentação:** `model_zoo/dinov2.md` e `model_zoo/dinov3.md`
- **Uso:** Baixar o `.pth` ou `.bin` e passar via `--model.ckpt_path`
- **DINOv3:** Por padrão espera delta weights relativos ao DINOv3. Para usar pesos absolutos, adicione `--model.delta_weights False`

---

## 7. Inferência Rápida

O arquivo `inference.ipynb` permite inferência e visualização com pesos pré-treinados auto-baixados. Útil para teste rápido sem treinar do zero.

---

## 8. Weights & Biases (wandb)

O projeto usa `wandb` para logging de experimentos. Para desativar, não passe flags de wandb. Para ativar:

```bash
wandb login
```

No Kaggle, configure a chave em **Settings → Secrets** como `WANDB_API_KEY`.

---

## 9. Boas Práticas para Edição no Kaggle

1. **Clone o repositório:**
   ```python
   !git clone https://github.com/tue-mps/eomt.git /kaggle/working/eomt
   ```

2. **Instale dependências:**
   ```python
   !pip install -r /kaggle/working/eomt/requirements.txt -q
   ```

3. **Não modifique arquivos upstream diretamente.** Crie um branch ou faça cópia local antes de editar:
   ```python
   import os
   os.chdir("/kaggle/working/eomt")
   ```

4. **Salve outputs em `/kaggle/working/`** para que persistam após o fim da sessão do notebook.

5. **Para repositórios privados ou forks**, use Personal Access Token do GitHub nas Secrets do Kaggle.

---

## 19. Problemas Conhecidos

| Problema | Solução |
|----------|---------|
| `CUDA out of memory` | Reduzir `--data.batch_size` para 1 ou 2 |
| Erro de importação | Verificar se `/kaggle/working/eomt` está no `sys.path` |
| Dataset não encontrado | Confirmar que os `.zip` estão em `--data.path` |
| Checkpoint não carrega | Usar `--model.load_ckpt_class_head False` para datasets com classes diferentes |

---

**Repositório oficial:** https://github.com/tue-mps/eomt.git  
**Contexto gerado para uso no ambiente Kaggle.**