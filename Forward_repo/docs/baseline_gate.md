# Puerta de paridad: `unroll714_sew` z24/Ncol

**Estado: paridad numérica T4 demostrada en sim300 y mar64; acreditación histórica de identidad todavía abierta.** La evaluación híbrida de septiembre usó un checkpoint intermedio del paso 400. Una copia recuperada del paso 1000, ejecutada con los paquetes extraídos, produjo métricas dentro de las tolerancias y estados casi idénticos a los históricos en ambos conjuntos. Falta el hash del checkpoint y de todas las entradas de la ejecución histórica. El inventario está en [`baseline_sew.json`](../manifests/baseline_sew.json), y los resultados nuevos en [`validation`](validation/README.md).

## Checkpoints y evidencia del paso 1000

| Archivo | SHA256 | `paso` | `args.steps` | Métricas finales |
|---|---|---:|---:|---|
| `unroll714_sew_ckpt.pt` (ruta histórica: `ops/modal/hybrid_oe_20260917/results/manifest.json`), entrada de la evaluación híbrida | `4b446240…e8bbe7a` | 400 | 1000 | No |
| `d400:WikiThesis_respaldo/salida/out_iasiang/trajsew/trajsew10_ckpt.pt` | `96833ee7…83fb7ea2296f` | 1000 | 1000 | Sí |
| `d400:WikiThesis_respaldo/salida/out_iasiang/trajz0/trajz0_24_ckpt.pt` | `1461c642…17fecb8309c35a` | 1000 | 1000 | Sí |

Los tres SHA256 se comprobaron contra bytes locales descargados. `torch.load(weights_only=True)` dio los pasos y campos mostrados. Las dos copias del paso 1000 proceden de evaluaciones `--solo-eval --resume /salida/out_iasiang/config/unroll714_sew_ckpt.pt`: `cadena_trajsew.sh` (ruta histórica: `model/physinv/closure_forward/experimentos/apae_loss_fisica/cadena_trajsew.sh`) y `cadena_trajz0.sh` (ruta histórica: `model/physinv/closure_forward/experimentos/apae_loss_fisica/cadena_trajz0.sh`). El registro de `trajz0` (ruta histórica: `experiments/2026-08_ablacion_prior/trajz0.log`) dice expresamente que cargó esa ruta en el paso 1000. El código de `exp18_unrolled.py` guarda un checkpoint final después de la evaluación, aun en `--solo-eval`.

Las dos copias recuperadas tienen exactamente los mismos diez tensores de la MLP: SHA256 `a2eceb6c4dfc6f6cdf8a0801c20e3cd315a14d91053e5b3a9578dfde0deb6a83` de sus bytes concatenados en orden de `state_dict`. Esos pesos difieren de los del checkpoint de paso 400. `R_SD` de `trajsew10` coincide con el del paso 400; el de `trajz0` difiere porque la evaluación usa otro arranque, por lo que no debe sustituirse `R_SD` de `trajsew10` por el de `trajz0`.

El SHA256 **del archivo de entrenamiento original** en el momento de publicar las tablas no se registró. Las copias de evaluación proporcionan pesos fuertemente vinculados al resultado histórico, pero no prueban la identidad byte a byte de aquel archivo original ni los hashes de sus demás entradas.

## Comparación simulada por escena

El `trajsew10_traj.npz` (ruta histórica: `experiments/2026-08_barrido_T/trajsew/trajsew10_traj.npz`) local tiene SHA256 `c58d9cf0adc70b617d9debc60dbb60460d6bb28077241093cdd612245128606b`, igual al de Drive. Sus filas y su `X_true` coinciden con el volcado histórico T4 (ruta histórica: `experiments/2026-08_barrido_T/panel8/sew_n4.npz`), este último tras conversión a `float32`. Entre `X_n[4]` de la trayectoria y `x_hat` histórico, el RMSE por elemento es `5.6836939705e-6`, el error absoluto medio `1.4158371705e-6` y el máximo `0.0001220703125`; el 85.934 % de los elementos son iguales en `float32`.

| Métrica | Histórico T4 | Trayectoria paso 1000, T4 |
|---|---:|---:|
| T (K) | 2.410006033340147 | 2.410006055393568 |
| T_skin (K) | 0.829344651406664 | 0.829343567833198 |
| H2O (dex) | 0.133130906634496 | 0.133130906699016 |
| CO2 (dex) | 0.004089053031135 | 0.004089053354215 |
| O3 (dex) | 0.107342846670086 | 0.107342836171429 |
| N2O (dex) | 0.061743548553341 | 0.061743548517773 |
| CH4 (dex) | 0.024812424170828 | 0.024812425295762 |

Esto acredita la **comparación numérica simulada** del candidato paso 1000 dentro de las tolerancias predefinidas. No es una nueva corrida completa de `exp24_baselines.py`; se recalcularon las métricas desde la trayectoria histórica guardada. La trayectoria incluye `z_n` y `r_n` por escena, pero no `R_SD`; ese tensor está en el checkpoint recuperado.

Después, la ejecución completa de T4 con los paquetes extraídos en Modal obtuvo gases medios `0.066223756224` dex, T `2.410006139703` K y T_skin `0.829343782229` K para las 300 escenas. Las diferencias absolutas históricas son `4.13e-10` dex, `1.06e-7` K y `8.69e-7` K, respectivamente. El [reporte](validation/sim300_T4.json) conserva las entradas y el entorno de esa corrida.

En contraste, la evaluación híbrida nueva (ruta histórica: `ops/modal/hybrid_oe_20260917/results/manifest.json`) usó el SHA256 `4b446240…e8bbe7a` del paso 400. Su etapa anterior a OE dio gases medios `0.070086740220` frente a `0.066223755812` dex históricos en sim300, y `0.072813692071` frente a `0.072248843665` dex históricos en mar64. T fue `2.598257` frente a `2.410006` K en sim300 y `2.368458` frente a `2.216883` K en mar64. Las filas y su orden coinciden en ambos conjuntos. El checkpoint intermedio es una causa plausible de la discrepancia, aunque el contraste controlado con las mismas entradas y ambos pesos todavía debe ejecutarse.

La auditoría local paralela del prior `prior_z_deep.pt` frente a los 30 000 latentes materializados de `prior_z_deep.npz` midió diferencia absoluta máxima `7.03e-6`, media `5.96e-7` y RMSE `7.80e-7`. Esta proximidad hace menos probable que la ruta de materialización del prior explique por sí sola el cambio de métricas; tampoco demuestra causalidad del checkpoint.

## Entradas de las 64 escenas IASI de mar

[`mar64_inputs.json`](../manifests/mar64_inputs.json) identifica por nombre, tamaño y SHA256 los **64 NPZ** de `valstates_full` (ruta histórica: `experiments/2026-07_apae_r2a/valstates_full/`) y los 14 catálogos NWP-SAF que aportan la fecha. `exp22_recon.py::carga_mar` lee `/datos/valstates_full/*.npz` en orden lexicográfico y `/datos/val64_geo.npz`. Los NPZ tienen las mismas 19 claves; el contrato principal incluye `T_levels(137)`, `Ncol(5,136)`, `P_lay(136)`, `nu(8461)`, `L_obs(8461)` y `L_obs_avg(8461)`, además de `T_skin`, `mu`, `psurf_hPa`, `cloud_frac`, `lat_fov` y `lon_fov`.

Las escenas reconstruyen exactamente `x_true`, `y_obs` en la banda 714–1250 cm⁻¹ y `P_lay` del volcado histórico de mar (ruta histórica: `experiments/2026-08_barrido_T/marplot/mar_sew_n4.npz`) en `float32`. El archivo `val64_geo.npz` se recuperó del volumen Modal con SHA256 de bytes `dd7909e261230f7fff3e4899db8e45595a9dffc339189f38ba6b43f58546d008`. Su contenido también pasó el digest semántico reconstruido de `lat`, `lon`, `doy`, `cloud`, `mu` y `nombres`. La ruta candidata de Drive permanece en el manifiesto, sin atribuirle la verificación de esos bytes.

La evaluación completa también requiere el dataset simulado de 30 000 estados, radiancias y ruido `iasiang`, `fechas_30k.npz`, `s_f`, APAE, prior, checkpoint, cinco LUT densas, `r_self.npz` y tablas del continuo. [`assets.json`](../manifests/assets.json) registra rutas y SHA256. Para algunos archivos grandes, la evidencia procede de metadatos `rclone lsjson --hash`; no debe describirse como verificación de bytes descargados. Los datos IASI y NWP-SAF de terceros permanecen en su almacenamiento autorizado; se publican solo esquemas, tamaños y hashes.

El directorio remoto `d400:WikiThesis_714/datos/valstates_full` contiene exactamente los 64 NPZ inventariados: `rclone lsjson --hash --max-depth 1` mostró tamaños y SHA256 coincidentes el 27 de septiembre de 2026. Es evidencia de **metadatos remotos**; [`mar64_fetch.py`](../pipelines/mar64_fetch.py) verifica los bytes después de cada descarga. En un volumen Modal fresco, `evaluate-mar64` obtiene solo las escenas ausentes en `/cache/data/valstates_full`, rechaza archivos existentes incorrectos y confirma todo el directorio antes de ejecutar el modelo. Con `rclone` configurado también puede ejecutarse localmente:

```powershell
python -m pipelines.mar64_fetch --scene-dir data_cache/data/valstates_full
```

El comprobador [`verify_mar64.py`](verify_mar64.py) verifica los 64 hashes, claves y formas, y opcionalmente los catálogos, `val64_geo.npz` recuperado y el volcado histórico. Esta invocación pasó localmente:

```powershell
python "thesis final/docs/verify_mar64.py" --scene-dir "experiments/2026-07_apae_r2a/valstates_full" --catalog-dir "data/atmospheres/nwpsaf_cams137_2025" --historical-dump "experiments/2026-08_barrido_T/marplot/mar_sew_n4.npz"
```

## Gate ejecutable

[`baseline_gate.py`](../pipelines/baseline_gate.py) compara sim300 y mar64 contra las métricas históricas de `baseline_sew.json`, con tolerancias absolutas fijas: gases medios ≤ **0.001 dex**, T ≤ **0.05 K** y T_skin ≤ **0.05 K**. Verifica filas ordenadas, z24/Ncol, banda `iasiang`, `se_corr=isrf` y T4. Acepta el JSON directo de `--metodo nuestro` o la etapa `pre_oe` de un JSON híbrido; no usa las métricas posteriores al refinamiento como resultado T4.

```powershell
python -m pipelines.report_adapter --run-record "ruta/evaluate_simulated_frozen300_T4.json" --out "ruta/sim300_T4_metrics.json"
python -m pipelines.baseline_gate --sim-json "ruta/sim300_T4_metrics.json" --sea-json "ruta/mar64_T4_metrics.json" --checkpoint "ruta/trajsew10_ckpt.pt" --run-manifest "ruta/evaluate_simulated_frozen300_T4.json" --run-manifest "ruta/evaluate_mar64_T4.json"
```

El informe separa `numeric_comparison` de `historical_identity`. La identidad exige el hash histórico del checkpoint, sus hashes de entrada y las identidades de la corrida candidata. También inspecciona con `torch.load(weights_only=True)` que el checkpoint tenga `paso=1000`, `args.steps=1000` y métricas finales. Sale con código 0 solo si pasan las métricas de ambos conjuntos y la identidad; 1 si falla alguna puerta y 2 para estructura inválida. **Las dos comparaciones numéricas pasan; el gate estricto sigue abierto** porque los hashes de la ejecución histórica original no quedaron registrados. El [informe mar64 por escena](validation/mar64_comparison.json) verifica los hashes de los resultados Modal y registra RMSE `5.7670e-6`, error absoluto medio `1.4385e-6` y máximo `9.1553e-5` entre las 64 recuperaciones nuevas e históricas, con `x_true`, `P_lay` y radiancias válidas exactamente iguales.

## Acreditación pendiente

La nueva corrida ya usó el checkpoint `trajsew10_ckpt.pt` y verificó todas las entradas disponibles antes de T4. Para acreditar identidad histórica falta recuperar un hash independiente del checkpoint de entrenamiento original y un inventario completo de sus entradas. Una comparación controlada con el checkpoint paso 400, manteniendo fijas las entradas actuales, podría medir cuánto de la discrepancia híbrida se debe a esos pesos; no es condición para la paridad numérica ya observada del modelo final.
