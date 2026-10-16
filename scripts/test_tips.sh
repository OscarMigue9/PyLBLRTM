#!/bin/bash
#SBATCH --job-name=test_tips
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --error=scripts/logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=cpu-cvail01
#SBATCH --account=oscar_miguel

# Test de las sumas de partición (TIPS) de todos los isotopólogos. Sin internet.
# Reporte en resultados_test_tips/<fecha_hora>/RESUMEN.md
#
# Enviar desde la raíz del repo:
#   sbatch scripts/test_tips.sh

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "AVISO: no existe .venv en $(pwd); se usa el python del sistema" >&2
fi

export PYTHONUNBUFFERED=1

echo "Nodo: $(hostname)   Inicio: $(date)"
python tests/tips_check.py "$@"
status=$?
echo "Fin: $(date)   Código de salida: $status"
exit $status
