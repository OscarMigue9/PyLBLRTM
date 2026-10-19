#!/bin/bash
#SBATCH --job-name=test_hapi
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --error=scripts/logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --partition=cpu-cvail01
#SBATCH --account=oscar_miguel

# Test de descarga HITRAN/HAPI: todas las moléculas e isotopólogos en 8–14 µm.
# No se detiene ante errores; al final deja el reporte en
#   resultados_test_hapi/<fecha_hora>/RESUMEN.md
#
# Enviar desde la raíz del repo (SLURM no crea la carpeta de logs):
#   mkdir -p scripts/logs
#   sbatch scripts/test_hapi.sh
#
# Argumentos extra se pasan al script de Python, por ejemplo:
#   sbatch scripts/test_hapi.sh --molecules H2O CO2 --iso first

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1

# Entorno de Python: .venv del repo, creado una sola vez con
#   python -m venv .venv && .venv/bin/python -m pip install -e .
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
else
    echo "AVISO: no existe .venv en $(pwd); se usa el python del sistema" >&2
fi

export PYTHONUNBUFFERED=1   # que el progreso aparezca en el .out en tiempo real

echo "Nodo: $(hostname)   Inicio: $(date)"
python --version
python -c "import hapi, numpy, scipy; print('hitran-api', getattr(hapi, 'HAPI_VERSION', '?'))"

python tests/hapi_download_check.py "$@"
status=$?

echo "Fin: $(date)   Código de salida: $status (1 = hubo errores, ver RESUMEN.md)"
exit $status
