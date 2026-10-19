#!/bin/bash
#SBATCH --job-name=test_forward
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --error=scripts/logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --partition=cpu-cvail01
#SBATCH --account=oscar_miguel

# Test del forward model con líneas reales de HITRAN2024 en 8–14 µm: firma
# standoff de cada molécula, atmósfera estándar, consistencia numérica y física.
# Usa la caché del último test de descarga (sbatch scripts/test_hapi.sh).
# Reporte en resultados_test_forward/<fecha_hora>/RESUMEN.md
#
# Necesita numba en el .venv (voigt="humlicek"):
#   .venv/bin/python -m pip install -e ".[fast]"
#
# Enviar desde la raíz del repo:
#   sbatch scripts/test_forward.sh

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "AVISO: no existe .venv en $(pwd); se usa el python del sistema" >&2
fi

export PYTHONUNBUFFERED=1

echo "Nodo: $(hostname)   CPUs: ${SLURM_CPUS_PER_TASK:-?}   Inicio: $(date)"
python tests/forward_check.py --n-jobs "${SLURM_CPUS_PER_TASK:-1}" "$@"
status=$?
echo "Fin: $(date)   Código de salida: $status"
exit $status
