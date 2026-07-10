# MicoViT (MicorizaeVision)

Repositorio de **código fuente** del pipeline de visión por tiles para cuantificación de colonización micorrícica.  
Remote configurado: `https://github.com/Nahzap/MicoViT.git`

> **Nota privada:** este README es la guía operativa para ti. No incluye documentación de investigación ni rutas a datos locales.

---

## Qué SÍ va en GitHub

| Incluido | Rutas típicas |
|----------|----------------|
| Código Python | `src/micorizae/`, `run.py`, `tools/`, `tests/`, `scripts/` |
| Config de esquema (sin datos) | `configs/*.yaml` |
| Dependencias | `requirements.txt`, `pyproject.toml` |
| Config local del pipeline | `config.py` |
| Este README | `README.md` |
| Metadatos git | `.gitignore`, `.gitattributes` |

## Qué NO debe subirse (bloqueado por `.gitignore`)

- **`Data/`** — imágenes, CSV de anotaciones, cualquier dato crudo o derivado.
- **`Docs/`** — documentación interna, planes, informes.
- **`cache/`**, **`manifests/`**, **`outputs/`**, **`models/`** — HDF5, embeddings `.npy`, checkpoints, parquet, logs de entrenamiento.
- **`.cursor/`**, **`.vscode/`**, `*.code-workspace` — entorno IDE.
- **Cualquier otro `*.md`** salvo este `README.md`.
- Secretos: `.env`, claves, credenciales.

Antes del push, revisa siempre:

```powershell
cd F:\MicorizaeVision
git status
git diff --cached --stat
```

Si aparece algo bajo `Data/`, `cache/`, `outputs/` o `Docs/`, **no hagas commit**.

---

## Clonar en otra máquina

```powershell
git clone https://github.com/Nahzap/MicoViT.git
cd MicoViT
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
# PyTorch con CUDA (ajusta índice según tu GPU):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Coloca tus datos **solo en local**, fuera de Git:

```
MicoViT/
  Data/          ← crear manualmente; ver configs/datasets.yaml
  manifests/     ← se genera con ingest
  cache/         ← se genera en runtime
  outputs/       ← corridas
  models/        ← checkpoints
```

En Windows puedes usar `scripts/setup_data_symlinks.ps1` si los datos viven en otro disco.

---

## Arranque mínimo (local)

```powershell
.\.venv\Scripts\Activate.ps1
python run.py ingest
python run.py tiles-index
```

Pipeline completo orquestado (6 pasos, salida en vivo a consola):

```powershell
python -u tools/run_micorizae_pipeline_from_zero.py
```

Comandos útiles vía CLI:

```powershell
python run.py --help
python run.py train-gate-am --help
python run.py train-stage2-pixel --help
```

Parámetros por defecto: **`config.py`** (editar solo en local; revisa qué expones si el repo deja de ser privado).

---

## Estructura del código

```
src/micorizae/
  phase_a_ingest/     Manifests desde Data/
  phase_b_tiling/     Índice de tiles
  phase_c_views/      Vistas / normalización GPU
  phase_d_stage1/     Gate (tile-level)
  phase_e_stage2/     Stage2 (subclases / píxel)
  phase_i_weakseg/    Weak segmentation helpers
  cli.py              Entrada Typer
  gate_runflow.py     Orquestación Gate
run.py                Wrapper CLI
tools/                Scripts de auditoría y pipelines
tests/                Smoke tests
configs/              Declaración de subsets (sin binarios)
```

---

## Checklist pre-push (primera subida)

1. **`git status`** — solo archivos de código/config esperados.
2. **`git diff`** — sin rutas absolutas personales ni comentarios que no quieras publicar en `config.py`.
3. Confirmar que **no hay** `.h5`, `.npy`, `.pt`, `.parquet`, imágenes ni CSV en el staging area.
4. Confirmar que **`Docs/`** no está trackeada.
5. Commit y push desde tu interfaz de GitHub (tú eliges mensaje y momento).

Comprobar qué ignoraría git un archivo:

```powershell
git check-ignore -v README.md Data/cache/stage2_pixel_mplus_v1.h5 Docs/
```

---

## Remote

```text
origin  https://github.com/Nahzap/MicoViT.git (fetch)
origin  https://github.com/Nahzap/MicoViT.git (push)
```

Si cambias de máquina:

```powershell
git remote set-url origin https://github.com/Nahzap/MicoViT.git
```

---

## Requisitos

- Python **3.11** (ver `pyproject.toml`)
- GPU NVIDIA recomendada para entrenamiento (CUDA)
- Espacio en disco local para `Data/` + `cache/` + `outputs/` (no versionados)

---

## Licencia

Definida en `pyproject.toml` (MIT). Ajusta si publicas el repo de forma abierta.
