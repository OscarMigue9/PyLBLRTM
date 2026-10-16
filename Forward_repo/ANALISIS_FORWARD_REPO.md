# Análisis completo de `Forward_repo`

> Fecha: 2026-10-05  
> Propósito: auditoría del repositorio como insumo para construir `pathtran`

---

## 1. Árbol general

```
Forward_repo/
├── configs/final.json                  ← configuración congelada del experimento
├── packages/
│   ├── photonix/                       ← librería CPU de transferencia radiativa
│   └── iasi_retrieval/                 ← red neuronal de inversión (APAE / prior / unrolled)
├── pipelines/                          ← orquestación de datos y entrenamiento
│   └── generation/                     ← construcción de LUTs, packs, síntesis de radianzas
├── manifests/                          ← inventario SHA256 de artefactos
├── frozen/                             ← datos pre-calculados que actúan como verdad
├── docs/                               ← provenance, migration notes, validation gates
└── tests/                              ← pruebas unitarias e integración
```

---

## 2. Qué hay en cada pieza

### 2.1 `packages/photonix` — el corazón de la física

Es una implementación CPU limpia del modelo AER de transferencia radiativa térmica para la banda 714–1250 cm⁻¹. Tiene tres capas bien separadas:

| Capa | Archivo(s) | Qué hace |
|---|---|---|
| Física de la atmósfera | `atmosphere.py` | Define presiones, temperaturas, VMRs (mol/mol), construye capas con densidades de columna |
| Espectroscopía | `spectroscopy/` | Perfiles Voigt, LBL óptico, continuo H₂O/CO₂, mezcla de líneas CO₂ (AER), tablas Q(T) de HITRAN |
| Transferencia radiativa | `transfer/` | Planck, integración upwelling/downwelling, convolución ILS gaussiana |
| Lookup table | `lut/cross_section.py` | Lee bloques SVD comprimidos (NPZ); retorna log₁₀(σ) en cm²/molécula |
| API pública | `forward.py` | `run_nadir_simulation()`, `at_sensor_radiance()` |

**Moléculas soportadas**: H₂O, CO₂, O₃, N₂O, CH₄, CO, O₂  
**Formatos de líneas**: HITRAN 160 cols, AER 100 cols  
**Perfil Voigt**: métodos Humlícek y Nelson-Simons  
**Continuo**: MT_CKD-3 style para H₂O y CO₂  

Constantes clave hardcodeadas:

```
BAND_MIN = 714.0 cm⁻¹       BAND_MAX = 1250.0 cm⁻¹
NU_FINE step = 0.001 cm⁻¹   NU_CHANNEL step = 0.25 cm⁻¹
T_AXIS: 170–330 K (20 puntos)
LOGP_AXIS: 1.4e-5–1.189 atm (145 puntos)
ILS FWHM = 0.5 cm⁻¹
DEFAULT_DIFFUSIVITY = 1.66
```

---

### 2.2 `packages/iasi_retrieval` — inversión ML (NO para pathtran directamente)

Inversión atmosférica con red neuronal para IASI. Vector de estado de 818 dimensiones:  
`T(137) + log₁₀(Ncol) × 5 gases × 136 capas + T_skin`

Componentes:

- **APAE** (`models.py`): autoencoder variacional 818→z24 + reconstrucción de T/U/D (6435 dim)  
- **Prior geográfico** (`prior.py`): red 4→256⁴→24 (lat, lon, doy, P_surf)  
- **Unrolled retrieval** (`unrolled.py`): 3 pasos compartidos con residual whitening y 30 ventanas de 25 cm⁻¹  
- **ForwardAdapter** (`forward_adapter.py`): estado 818-dim → T/U/D usando 5 LUTs densas NPY (20×145×630001)

Este módulo es ML puro y está fuertemente acoplado a datos IASI. Se puede estudiar como referencia de diseño pero no se migra tal cual.

---

### 2.3 `pipelines/` — orquestación y generación de datos

- **`assets.py`**: sistema de artefactos con SHA256 y manifiestos (muy reutilizable)  
- **`generation/lut.py`**: construcción de bloques SVD (estructura U, V, mu, a, b, NT, NP, r, dnu)  
- **`generation/gpu_pack.py`**: reconstrucción del pack espectroscópico histórico (lineas.npz, qtab.npz, cont.npz)  
- **`generation/raw.py`**: extracción de estados atmosféricos de CAMS NetCDF → columnas de gas  
- **`generation/synthesis.py`**: ensamblaje de radianzas y ruido  
- **`modal_final.py`**: entrenamiento en Modal GPU (A100-80GB, torch 2.7.1)  
- **`provenance.py`**: snapshot de código fuente (hash de todos los archivos relevantes)

---

### 2.4 `configs/final.json` — configuración congelada del experimento `z24/Ncol`

Define el nombre del experimento, la receta física, grilla espectral, estructura de LUT, semillas de entrenamiento, e hiperparámetros de APAE/prior/unrolled. Es el punto único de verdad para reproducir el experimento de tesis.

---

### 2.5 `frozen/` — datos pre-calculados

| Archivo | Contenido |
|---|---|
| `gpu_pack/lineas.npz` | Líneas espectrales bundleadas (formato legado GPU) |
| `gpu_pack/qtab.npz` | Tablas de función de partición |
| `gpu_pack/cont.npz` | Tablas de continuo |
| `split_30k_seed42.npz` | Split train/val/test (24k/3k/3k), semilla 42 |
| `nuestro_n300.json` | 300 escenas de prueba seleccionadas |
| `val64_geo.npz` | 64 escenas oceánicas (MAR-64) |
| `record_ids.npy` | IDs de registros ECOSTRESS |

---

### 2.6 `docs/` — documentación de provenance y validación

Documentos clave que registran la historia del experimento, sin código ejecutable pero con tolerancias y gates de aceptación numérica:

- `FORWARD_MIGRATION.md`: diferencias entre receta histórica GPU y receta CPU (Photonix)
- `TRAINING_PROVENANCE.md`: historia de checkpoints APAE/prior
- `DATA_PROVENANCE.md`: cadena de transformaciones de datos
- `REPRODUCTION_PLAN.md`: gates de aceptación (RMSE temperatura ≤0.05 K, gases ≤0.001 dex)
- `baseline_gate.md`: validación numérica del baseline

---

## 3. Qué necesitamos para `pathtran`

### 3.1 Módulos directamente reutilizables (migrar con adaptación)

| Prioridad | Módulo | Archivos | Adaptación necesaria |
|---|---|---|---|
| **Alta** | Espectroscopía LBL | `spectroscopy/absorption.py`, `voigt.py`, `species.py` | Desacoplar de constantes IASI-específicas |
| **Alta** | Perfil atmosférico | `atmosphere.py` | Generalizar beyond 137 niveles; quitar hardcodes IASI |
| **Alta** | Transferencia radiativa | `transfer/planck.py`, `integrator.py`, `ils.py` | Parametrizar banda |
| **Alta** | Constantes físicas | `constants.py` | Mover a `pathtran.constants` directo |
| **Media** | Parsing HITRAN | `spectroscopy/hitran.py` | Integrar con HAPI (ya tenemos `pylblrtm`) |
| **Media** | Continuo H₂O/CO₂ | `spectroscopy/continuum.py` | Verificar convención MT_CKD vs lo que usa HAPI |
| **Media** | Lector de LUT SVD | `lut/cross_section.py` | Generalizar a cualquier banda/receta |
| **Baja** | Mezcla de líneas CO₂ | `spectroscopy/aer.py` | Solo necesario si se quiere alta precisión en 15 µm |

### 3.2 Arquitectura e ideas para replicar (no código directo)

| Idea | Origen | Por qué vale |
|---|---|---|
| Sistema de manifiestos SHA256 | `pipelines/assets.py` | Reproducibilidad sin ambigüedad; cada artefacto tiene hash |
| Bloques SVD para LUTs | `generation/lut.py`, `lut/cross_section.py` | Compresión ~10× de la tabla de secciones transversales |
| Provenance snapshot automático | `pipelines/provenance.py` | Hash de todo el código fuente en cada corrida |
| Separación fina/canales en grilla | `forward.py` | NU_FINE (física) vs NU_CHANNEL (instrumento) bien separados |
| Validación `_validation.py` | `_validation.py` | Contratos de entrada explícitos, errores claros |
| Gates numéricos de aceptación | `docs/REPRODUCTION_PLAN.md` | Define cuándo "pasa" la reproducción, no solo si corre |

### 3.3 Qué NO llevamos a pathtran

| Elemento | Por qué no |
|---|---|
| `iasi_retrieval/` completo | Es ML específico de IASI; pathtran es solo forward |
| Pesos del APAE/prior | No son parte de una librería de transferencia radiativa |
| `modal_final.py` | Infraestructura de entrenamiento en la nube; fuera de alcance |
| `frozen/split_30k*.npz` | Datos de entrenamiento privados/de tesis |
| `pipelines/final.py` | Orquestador de entrenamiento ML |
| Manifiestos de ECOSTRESS/CAMS | Datos de entrenamiento específicos del experimento |

---

## 4. Propuestas de lo que podemos construir

### 4.1 Integración inmediata: núcleo forward CPU

Tomar `photonix` casi tal cual y convertirlo en el módulo `pathtran.forward`:

```
pathtran/
├── forward.py          ← run_nadir_simulation() (de photonix/forward.py)
├── atmosphere.py       ← AtmosphericProfile, Layer
├── constants.py        ← constantes físicas unificadas
├── spectroscopy/
│   ├── species.py      ← definiciones de moléculas
│   ├── hitran.py       ← parser HITRAN (+ puente a HAPI)
│   ├── absorption.py   ← optical depth LBL
│   ├── voigt.py        ← perfil de Voigt
│   └── continuum.py    ← continuo H₂O/CO₂
└── transfer/
    ├── planck.py
    ├── integrator.py
    └── ils.py
```

### 4.2 Puente HAPI (medio plazo)

`hitran.py` actual parsea archivos `.par` directo. Tenemos `pylblrtm` que ya usa HAPI. La propuesta es:

1. `pathtran.spectroscopy.hitran.from_hapi(molecule, iso, nu_min, nu_max)` → retorna `LineList` compatible con lo que espera `absorption.py`
2. El usuario puede usar HAPI como backend de líneas sin cambiar la cadena de transferencia radiativa

### 4.3 Soporte de banda configurable

Las constantes `BAND_MIN=714`, `BAND_MAX=1250` están hardcodeadas en `forward.py`. Propuesta:

- Parámetro `band=(nu_min, nu_max)` en `run_nadir_simulation()`
- Validación automática de que la LUT cubre la banda pedida
- Allows usar pathtran para otras bandas (MWIR, SWIR)

### 4.4 Backend LUT SVD público

El sistema de LUTs SVD es muy bueno pero los datos están pinneados a Google Drive privado. Propuesta:

- Definir un formato público de LUT SVD (spec del NPZ: keys U, V, mu, a, b, NT, NP, r, dnu)
- Script de construcción que corra contra cualquier fuente HITRAN
- Documentar cómo un usuario genera su propia LUT

### 4.5 Sistema de manifiestos ligero para pathtran

Adaptar `pipelines/assets.py` en una clase `pathtran.data.Catalog`:

- Descarga de datos de referencia (líneas HITRAN, tablas de continuo) con verificación SHA256
- Cache local en `~/.pathtran/cache/`
- Sin dependencia de infraestructura Modal/Google Drive

### 4.6 Suite de validación física

Basado en `docs/REPRODUCTION_PLAN.md` y los tests actuales, proponer una suite:

- Test de cuerpo negro: radiance de atmósfera isotérmica = Planck(T)
- Test de Beer-Lambert: gas único con VMR constante vs solución analítica
- Test de conservación de energía: transmittance + absortance = 1 (capas optically thin)
- Benchmark vs LBLRTM o Py4CATS para un perfil estándar (US Standard Atmosphere)

---

## 5. Problemas conocidos a resolver antes de migrar

| Problema | Origen | Impacto |
|---|---|---|
| Constantes del historial GPU vs CPU difieren | `torch_legacy/` vs `photonix/` | Parity test puede fallar si se mezclan |
| CH4 LUT: RMS falla en canal terminal | `docs/baseline_gate.md` | Posible error sistemático en CH₄ cerca de 1250 cm⁻¹ |
| Ruido rebuild: 2 celdas float32 difieren | `docs/` | Irrelevante para pathtran (es ruido de sensor) |
| Seed del prior histórico no registrada | `docs/TRAINING_PROVENANCE.md` | Solo afecta reproducibilidad del prior ML |
| `hitran.py` asume AER 100-col o HITRAN 160-col, no HAPI format | `spectroscopy/hitran.py` | Requiere puente explícito con pylblrtm/HAPI |

---

## 6. Resumen ejecutivo

**Lo más valioso de Forward_repo para pathtran**:

1. `photonix` es un forward CPU limpio, bien estructurado, con validación de entradas y separación clara de capas físicas. Es la base directa de `pathtran.forward`.
2. El esquema de LUTs SVD es eficiente y generalizable a cualquier banda o sensor.
3. Los docs de provenance y los gates numéricos establecen una cultura de validación que pathtran debe heredar.
4. La separación NU_FINE / NU_CHANNEL es un patrón que debemos preservar.

**Lo que hay que dejar atrás**:
El módulo `iasi_retrieval` y los pipelines de ML son específicos del experimento de tesis. No son librería, son experimento.

**Próximo paso sugerido**:
Tomar `photonix` como punto de partida, generalizar la configuración de banda, y construir el puente HAPI en `spectroscopy/hitran.py`. Con eso tenemos el 80% del valor de Forward_repo integrado en pathtran.
