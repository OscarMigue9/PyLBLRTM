#!/bin/bash
#SBATCH --job-name=run_notebook
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --error=scripts/logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --partition=cpu-cvail01
#SBATCH --account=oscar_miguel

# Ejecuta un notebook completo y guarda una copia con los resultados
# (<nombre>_ejecutado.ipynb) junto al original, que queda sin tocar.
#
# Enviar desde la raíz del repo:
#   sbatch scripts/run_notebook.sh                                  # el de 8–14 µm
#   sbatch scripts/run_notebook.sh notebooks/otro.ipynb
#
# Necesita en el .venv:  pip install -e ".[plot,fast]" ipykernel nbconvert
# y el kernel:           python -m ipykernel install --user --name pylblrtm

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "AVISO: no existe .venv en $(pwd); se usa el python del sistema" >&2
fi

export PYTHONUNBUFFERED=1
NB="${1:-notebooks/transmitancia_8_14um.ipynb}"
OUT="$(basename "${NB%.ipynb}")_ejecutado"

echo "Nodo: $(hostname)   CPUs: ${SLURM_CPUS_PER_TASK:-?}   Inicio: $(date)"
echo "Notebook: $NB  →  $(dirname "$NB")/$OUT.ipynb"
jupyter nbconvert --to notebook --execute "$NB" --output "$OUT" \
    --ExecutePreprocessor.timeout=1200
status=$?
echo "Fin: $(date)   Código de salida: $status"
exit $status
