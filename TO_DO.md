# TO DO - Auditoria de la metodologia energetica

## Objetivo

Documentar el problema detectado en la medida de potencia/eficiencia del proyecto, dejar evidencia concreta de por que no podemos defender todavia todas las metricas energeticas, y proponer lineas de investigacion y soluciones plausibles para corregirlo.

Este fichero resume:

- Que esta funcionando bien.
- Que esta fallando o es metodologicamente debil.
- Que evidencia tenemos a dia de hoy.
- Que hipotesis son mas plausibles.
- Que cambios conviene probar.
- Como validar que una solucion realmente mejora la metodologia.


## Estado actual

### 1. Lo que parece correcto

- El rendimiento absoluto de los benchmarks es estable entre la campana historica y la actual.
- La separacion entre `BW peak` y `BW sustained` corrige una ambiguedad real del pipeline anterior.
- La comparacion A100 vs RTX5000 tiene sentido a nivel arquitectonico:
  - A100 destaca mucho mas en FP64.
  - A100 destaca mucho mas en GEMM con TF32.
  - BW sostenido es claramente superior en A100.

### 2. Lo que no parece defendible todavia

- La eficiencia energetica no es igual de fiable en todos los tests.
- Hay varios valores de eficiencia que implican potencias demasiado bajas para la carga observada.
- El problema no parece un fallo del benchmark de rendimiento, sino un problema de metodologia de medida de potencia.


## Evidencia principal

## A. La anomalia ya existia antes de los cambios recientes

Se comparo la campana actual con el historico:

- Archivo historico:
  - `results/tfg_results_20260306_174558.tgz`

Conclusiones:

- El rendimiento entre campanas es casi identico.
- La anomalia energetica ya estaba presente en marzo.
- Los cambios nuevos no parecen haber roto el benchmark.
- Lo que si han hecho es exponer mejor el problema al medir por subprueba concreta.

### Rendimiento historico vs actual

Los cambios son muy pequenos en rendimiento:

- `BW peak`: practicamente igual.
- `Compute FP32 peak`: practicamente igual.
- `Compute FP64 peak`: practicamente igual.
- `GEMM TF32=0`: practicamente igual.
- `GEMM TF32=1`: cambio pequeno, pero del orden esperado.

Interpretacion:

- El codigo CUDA y el pipeline de rendimiento parecen estables.

## B. La eficiencia vieja ya tenia sintomas raros

En la campana historica ya aparecian eficiencias sospechosas, por ejemplo:

- A100 `GEMM TF32=1 max` daba una eficiencia muy alta para la potencia observada.
- En el pipeline antiguo, una misma potencia por modo se reutilizaba para varias subpruebas:
  - `Compute FP32` y `Compute FP64`
  - `GEMM TF32=0` y `GEMM TF32=1`

Eso confirma que:

- Habia un problema metodologico real en la version anterior.
- La campana antigua no puede tomarse como referencia energetica "limpia".

## C. Los logs crudos muestran incoherencias internas

En los logs historicos de A100 aparecieron muestras del tipo:

- `utilizacion muy baja` + `potencia muy alta` + `SM clock alto`
- `utilizacion muy alta` + `potencia muy baja` + `SM clock alto`

Ejemplos detectados:

- muestras con `util <= 5%` y `power >= 150 W`
- muestras con `util >= 80%`, `SM clock >= 1400 MHz` y `power < 80 W`

Eso apunta a que la telemetria de `nvidia-smi` no esta perfectamente sincronizada entre columnas, o que hay latencia/desfase temporal entre:

- `power.draw`
- `clocks.sm`
- `utilization`


## Diagnostico tecnico actual

### Hipotesis principal

La metodologia actual sobreestima la eficiencia en algunos casos porque:

- usa `nvidia-smi` con muestreo cada `10 ms`
- define "actividad" usando principalmente `SM clock`
- algunos kernels seleccionados como "peak" son demasiado cortos
- el logger captura bastante tiempo antes/despues del trabajo util
- la GPU puede mantener clocks altos durante periodos donde la potencia ya cae

Resultado:

- se consideran "activas" muestras que no representan trabajo util sostenido
- la potencia media activa queda infraestimada
- la eficiencia `Perf / Power` queda inflada


## Casos mas problematicos

### 1. BW peak

Especialmente delicado porque el caso elegido puede ser muy pequeno.

En RTX5000 actual:

- `BW peak` se selecciona en `1 MB`
- duracion aproximada del kernel: `0.712 ms`
- logger: `10 ms`

Esto significa que la medida energetica esta muy contaminada por:

- overhead de lanzamiento
- cola antes/despues del kernel
- dinamica de clocks

Conclusion:

- `BW peak` no es una buena metrica energetica con esta metodologia.

### 2. Compute FP32 peak

En RTX5000 actual:

- duracion aproximada: `0.549 ms`
- el log dura mucho mas que el trabajo util real

Se observaron muchas muestras activas con potencia baja.

Conclusion:

- `Compute FP32 peak` no es defendible como metrica energetica con el pipeline actual.

### 3. Compute FP64 peak

En RTX5000 actual:

- duracion aproximada: `20.595 ms`

Es mejor que FP32 peak, pero sigue estando en una zona delicada.

Conclusion:

- puede usarse con cautela, pero no es de las metricas energeticas mas robustas.

### 4. GEMM TF32=0 y GEMM TF32=1

En RTX5000 actual:

- duracion aproximada por caso: `~1965 ms`

Esto ya esta mucho mas cerca de algo medible.

Aun asi:

- siguen apareciendo bastantes muestras activas con potencias bajas
- parece haber cola o transicion dentro del intervalo considerado activo

Conclusion:

- son las metricas energeticas mas prometedoras del pipeline actual
- aun conviene refinarlas

### 5. BW sustained

Es mucho mas defendible que `BW peak` porque:

- la transferencia es grande
- el trabajo util es largo
- la potencia se estabiliza mejor

Conclusion:

- es de las metricas energeticas mas utiles ahora mismo


## Evidencia concreta de la campana RTX5000 actual

### Duraciones aproximadas de los casos seleccionados

- `BW peak`: `0.712 ms`
- `BW sustained`: `4621.823 ms`
- `Compute FP32 peak`: `0.549 ms`
- `Compute FP64 peak`: `20.595 ms`
- `GEMM TF32=0 max`: `1965.024 ms`
- `GEMM TF32=1 max`: `1965.885 ms`

### Implicacion metodologica

Con `sample_ms = 10`:

- `BW peak` y `Compute FP32 peak` son demasiado cortos
- `Compute FP64 peak` esta en una zona dudosa
- `BW sustained` y `GEMM` son mucho mas adecuados para medir potencia

### Observaciones en RTX5000 actual

Del analisis de los logs crudos:

- `Compute FP32 peak`: gran parte de las muestras activas tienen potencia baja
- `GEMM TF32=0`: existe una proporcion relevante de muestras activas con potencia baja
- `GEMM TF32=1`: existe una proporcion relevante de muestras activas con potencia baja

Interpretacion:

- el filtro por `SM clock` no aísla bien el tramo de trabajo util sostenido


## Pregunta clave a resolver

La pregunta no es:

- "El benchmark esta roto?"

La pregunta correcta es:

- "La metodologia de medida de potencia esta capturando realmente el trabajo util del caso seleccionado?"

Ahora mismo, la respuesta parece ser:

- si, parcialmente, para `BW sustained` y `GEMM`
- no, o no de forma fiable, para `BW peak` y `Compute FP32 peak`


## Soluciones plausibles

## Opcion 1. Alargar artificialmente los casos usados para energia

Esta es la solucion mas plausible y mas facil de defender.

Idea:

- no medir energia sobre un solo kernel "peak"
- repetir internamente el mismo caso muchas veces dentro de una sola ejecucion larga
- medir potencia durante una ventana suficientemente larga y estable

Ejemplos:

- repetir `Compute FP32 peak` durante 1-3 segundos continuos
- repetir `BW peak` durante 1-3 segundos continuos
- mantener `GEMM` como esta o alargar ligeramente si hace falta

Ventajas:

- mejora mucho la observabilidad con `nvidia-smi`
- reduce el peso del overhead de lanzamiento
- reduce el efecto de transiciones de clocks
- hace las eficiencias mucho mas defendibles

Esta es, probablemente, la mejor linea de trabajo.


## Opcion 2. Medir energia-sobre-tiempo con ventana estable

Idea:

- en vez de coger todas las muestras activas del log completo
- detectar una ventana central estable
- ignorar calentamiento y enfriamiento

Como podria hacerse:

- descartar el 10-20% inicial y final del tramo activo
- o buscar una meseta de potencia/reloj con baja varianza

Ventajas:

- evita que colas de entrada/salida sesguen la potencia media

Riesgos:

- mas heuristico
- mas dificil de justificar si no se documenta muy bien


## Opcion 3. Redefinir que es "activo"

Ahora mismo, la actividad se filtra por `SM clock`.

Problema:

- clocks altos no implican necesariamente trabajo util sostenido

Alternativas:

- combinar `SM clock` con `power.draw`
- combinar `SM clock` con `utilization.gpu` si se registra de forma fiable
- usar un umbral mas estricto

Ejemplos:

- activo si `SM clock >= x` y `power.draw >= y`
- activo si `SM clock >= x` y la muestra esta dentro de una ventana temporal central

Ventajas:

- puede limpiar parte del ruido

Riesgos:

- sigue dependiendo de `nvidia-smi`
- puede convertirse en un filtro demasiado "a medida"


## Opcion 4. Cambiar la fuente de telemetria

Investigar alternativas a `nvidia-smi`, por ejemplo:

- NVML directa desde codigo
- Nsight con metricas energeticas si el entorno lo soporta
- instrumentacion externa

Ventajas:

- podria dar una medida mas controlada

Riesgos:

- mas complejidad
- mayor coste de implementacion
- puede no ser viable dentro del tiempo del TFG

Para un TFG, esta opcion es interesante como linea futura, pero no necesariamente la mejor para cerrar esta fase.


## Opcion 5. Descartar algunas metricas energeticas y defender solo las robustas

Idea:

- no todas las metricas tienen por que tener la misma calidad metodologica

Se podria defender:

- rendimiento para todos los tests
- energia/eficiencia solo para:
  - `BW sustained`
  - `GEMM TF32=0`
  - `GEMM TF32=1`
  - quizas `Compute FP64`, con cautela

Y dejar fuera:

- `BW peak` energetico
- `Compute FP32 peak` energetico

Ventajas:

- honesto metodologicamente
- reduce el riesgo de sobreinterpretar datos malos

Riesgos:

- reduce la cobertura de la parte energetica
- obliga a explicarlo claramente en la memoria


## Recomendacion priorizada

Orden recomendado de investigacion:

### Prioridad 1

Alargar los casos usados en energia.

Objetivo:

- que cada caso dure al menos del orden de `1 s`
- idealmente, varios cientos de muestras o un regimen claramente estable

### Prioridad 2

Recalcular la potencia usando una ventana estable del tramo activo.

### Prioridad 3

Revisar la definicion de "activo" para no depender solo del `SM clock`.

### Prioridad 4

Si sigue habiendo incoherencias fuertes, investigar otra fuente de telemetria.


## Plan de investigacion sugerido

### Paso 1. Crear una version "long-run" de cada subprueba energetica

Anadir un modo energetico que:

- repita internamente el mismo kernel/caso
- ejecute durante un tiempo objetivo fijo
- escriba un CSV de rendimiento agregado y tiempo total

Posible enfoque:

- `--energy-duration-ms 2000`
- `--energy-min-repeats N`

### Paso 2. Volver a lanzar una campana corta de validacion

Comparar:

- rendimiento absoluto
- potencia media
- estabilidad de clocks
- porcentaje de muestras activas

### Paso 3. Inspeccionar si la eficiencia converge

Senal positiva:

- la eficiencia deja de depender tanto de colas o muestras de baja potencia
- la potencia media se parece mas a lo esperable fisicamente

### Paso 4. Repetir varias veces

Hacer al menos 3 repeticiones por entorno para ver:

- dispersion
- sensibilidad de la medida


## Criterios para dar el problema por resuelto

Una solucion deberia cumplir, idealmente:

- el rendimiento absoluto sigue estable
- las potencias medias son fisicamente creibles
- los casos cortos dejan de producir eficiencias absurdamente altas
- las metricas energeticas varian poco entre repeticiones
- la metodologia puede explicarse de forma clara en la memoria


## Que revisar en la memoria del TFG

Si este problema no se corrige del todo, la memoria deberia decir explicitamente:

- que la potencia se mide con `nvidia-smi`
- que el alcance de potencia es `gpu_board`, no nodo completo
- que las cargas muy breves son dificiles de medir con esta metodologia
- que no todas las metricas energeticas tienen la misma robustez
- que las metricas de rendimiento son mas solidas que las energeticas en el estado actual


## Archivos relevantes para seguir investigando

- `scripts/run_campaign.sh`
- `scripts/energy_active_summary.py`
- `scripts/perf_targets.py`
- `scripts/power_logger.sh`
- `src/main.cu`
- `src/bw_bench.cu`
- `src/compute_bench.cu`
- `src/gemm_bench.cu`

Resultados y evidencias:

- `results/tfg_results_20260306_174558.tgz`
- `results/compare/summary_compare.csv`
- `results/compare/environment_compare.csv`
- `results/compare/methodology_notes.txt`

Resultados actuales externos copiados para auditoria:

- `D:/Descargas/resultados/summary_compare.csv`
- `D:/Descargas/resultados/environment_compare.csv`
- `D:/Descargas/resultados/methodology_notes.txt`
- `D:/Descargas/resultados/rtx5000/energy/*`


## Decision provisional recomendada

Hasta no corregir esta parte, la posicion mas honesta y defendible es:

- usar rendimiento absoluto y comparativo como resultado principal
- usar eficiencia solo con cautela
- no sacar conclusiones fuertes a partir de `BW peak` energetico ni `Compute FP32 peak` energetico
- presentar esta auditoria como parte del valor tecnico del trabajo


## Nueva tarea - Benchmark de control no basado en GEMM

## Contexto

En reunion con el profesor se plantea introducir un benchmark adicional con una libreria fiable que:

- no sea GEMM
- no este centrado en multiplicacion de matrices densas
- ayude a que la comparacion entre A100 y RTX5000 no quede tan dominada por una ventaja "obvia" de arquitectura
- sirva como carga de control o contraste frente a los casos actuales

Interpretacion razonable de esta peticion:

- no se trata de "forzar" artificialmente que la RTX5000 gane
- se trata de introducir una carga representativa donde el resultado dependa menos de Tensor Cores / FP64 / cuBLAS y mas de otros aspectos del hardware
- esto hace la memoria mas equilibrada y metodologicamente mas fuerte


## Que sentido tiene esta peticion

La peticion del profesor tiene sentido por varios motivos:

- GEMM con cuBLAS, y especialmente TF32 o FP64 en A100, favorece mucho a la A100 por diseno arquitectonico
- eso no invalida el benchmark actual, pero si hace que una parte de la narrativa sea demasiado esperable
- un benchmark adicional no matricial puede funcionar como "control workload"
- permite mostrar que no todo en la comparacion depende de las mismas unidades funcionales
- mejora la calidad academica del trabajo porque introduces diversidad de patrones de acceso y computacion

En otras palabras:

- GEMM responde bien a la pregunta "como de buena es cada GPU en algebra lineal densa altamente optimizada?"
- hace falta otra carga para responder "como se comportan en otra familia de workloads CUDA reales?"


## Candidatos evaluados

### Opcion A. cuFFT

Biblioteca:

- `cuFFT`, biblioteca oficial de CUDA para transformadas rapidas de Fourier

Por que encaja bien:

- es oficial de NVIDIA
- viene dentro del ecosistema CUDA
- no es GEMM ni algebra lineal densa
- es muy conocida en HPC, procesado de senal, imagen, simulacion, espectroscopia, etc.
- soporta planificacion flexible con `cufftCreate`, `cufftMakePlan*` y ejecucion con `cufftExec*`
- permite ejecutar casos batched, lo que ayuda a construir pruebas largas y medibles energeticamente

Ventajas metodologicas:

- reduce el sesgo especifico de Tensor Cores / cuBLAS
- sigue siendo una carga de biblioteca real, no un microkernel artificial
- es bibliograficamente mas facil de defender que un benchmark casero

Limitaciones:

- la A100 seguira ganando en muchos casos
- no garantiza una comparacion "igualada"
- el rendimiento depende bastante del tamano, dimensionalidad, batch y tipo de datos

Veredicto:

- es la mejor opcion si solo se va a anadir una biblioteca no matricial al proyecto

### Opcion B. CUB DeviceRadixSort o DeviceReduce

Biblioteca:

- `CUB`, libreria oficial de primitivas paralelas de NVIDIA dentro de CCCL

Posibles primitivas:

- `cub::DeviceRadixSort`
- `cub::DeviceReduce`
- `cub::DeviceScan`

Por que encaja bien:

- es oficial de NVIDIA
- no usa matrices
- representa primitivas muy fundamentales de movimiento de datos y paralelismo
- puede dar una comparacion menos sesgada por unidades de GEMM

Ventajas:

- muy buena como carga de control
- mas "general-purpose" que GEMM
- mas ligada a memoria, reordenacion y throughput de primitivas

Limitaciones:

- bibliograficamente puede ser mas dificil de presentar que cuFFT si se usa como benchmark principal
- el resultado puede depender mucho del tamano y tipo del dato
- es menos "aplicacion final" y mas "primitive benchmark"

Veredicto:

- muy buena opcion secundaria o complementaria
- si el tiempo lo permite, seria excelente como benchmark de control adicional

### Opcion C. NPP

Biblioteca:

- `NPP`, libreria oficial de procesado 2D de imagen y senal

Ventajas:

- oficial
- no matricial
- ligada a procesado de imagen y senal

Limitaciones:

- es mas de dominio concreto
- elegir una operacion representativa y justa es mas delicado
- es menos natural que cuFFT para un benchmark compacto y reproducible dentro del repo actual

Veredicto:

- opcion valida, pero no la recomendaria como primera eleccion

### Opcion D. cuRAND

Biblioteca:

- `cuRAND`, libreria oficial de generacion de numeros aleatorios

Ventajas:

- oficial
- no matricial
- util para Monte Carlo u otras cargas estocasticas

Limitaciones:

- medir solo RNG puede aportar menos valor interpretativo
- si se mezcla con un kernel propio, parte del benchmark deja de estar en la libreria
- como benchmark principal es menos redondo que cuFFT

Veredicto:

- interesante para trabajos futuros, pero no la opcion principal recomendada

### Opcion E. cuSPARSE

Biblioteca:

- `cuSPARSE`

Ventajas:

- oficial
- mas cercana a cargas HPC reales

Limitaciones:

- sigue siendo una biblioteca de matrices
- aunque sea dispersa y no GEMM, va contra la peticion explicita de evitar matrices

Veredicto:

- descartada como primera opcion para esta peticion concreta


## Veredicto final recomendado

### Si solo se añade un benchmark

La mejor recomendacion es:

- anadir un benchmark con `cuFFT`

### Si se pueden añadir dos

La mejor combinacion seria:

- `cuFFT` como benchmark principal no matricial
- `CUB DeviceRadixSort` o `CUB DeviceReduce` como benchmark de control complementario

Interpretacion:

- `cuFFT` aporta una carga real de biblioteca, bien conocida y facil de justificar
- `CUB` aporta una carga aun menos sesgada por GEMM y muy util para contraste


## Por que cuFFT es la mejor opcion principal

### 1. Es una biblioteca oficial, madura y conocida

Esto es importante para la memoria del TFG.

Permite decir:

- no se ha escogido un benchmark inventado ad hoc
- se usa una libreria estandar del ecosistema CUDA
- el trabajo compara familias de workloads distintas dentro del mismo vendor stack

### 2. No esta centrada en algebra lineal densa

Eso ataca directamente el problema que ha detectado el profesor.

### 3. Sigue siendo HPC y cientificamente relevante

FFT tiene sentido en:

- procesamiento de senal
- procesamiento de imagen
- metodos espectrales
- simulacion cientifica
- analisis de frecuencias

### 4. Es implementable sin reestructurar medio repo

Se puede integrar con un coste razonable.

### 5. Permite casos largos

Esto es clave por el problema energetico ya detectado.

Con `batch` suficientemente grande y tamanos adecuados:

- la ejecucion puede durar cientos de milisegundos o segundos
- eso la hace mucho mas medible con `nvidia-smi`


## Hipotesis de trabajo sobre el comportamiento A100 vs RTX5000

Esto es una inferencia tecnica, no una garantia:

- en `cuFFT`, la A100 probablemente seguira ganando
- pero la diferencia deberia ser menos "obvia" y menos extrema que en:
  - `GEMM TF32`
  - `FP64 GEMM-like workloads`
  - cargas claramente favorecidas por Tensor Cores o por el perfil HPC de A100

La idea no es buscar un benchmark donde RTX5000 "gane", sino:

- uno donde la comparacion sea menos trivial
- uno que introduzca otra dimension de analisis


## Como deberia implementarse en este repo

## Opcion recomendada - `fft_bench.cu`

Crear un nuevo benchmark:

- `src/fft_bench.cu`

Integracion minima:

- enlazar `CUDA::cufft` en `CMakeLists.txt`
- ampliar `src/main.cu` con un nuevo modo:
  - `--mode fft`
- generar un CSV:
  - `results/<env>/baseline/fft.csv`
- generar grafica:
  - `scripts/plot_fft.py`

### API recomendada

Usar:

- `cufftCreate()`
- `cufftMakePlanMany()` o `cufftPlanMany()`
- `cufftExecC2C()` o `cufftExecR2C()`

### Configuracion recomendada

Primera implementacion sugerida:

- caso principal: `C2C` en precision simple
- 1D batched
- tamanos potencia de 2 y algun tamano no potencia de 2

Por ejemplo:

- `N = 2^18`, `2^20`, `2^22`
- `batch` ajustado para ocupar bastante memoria pero sin acercarse demasiado al limite de la RTX5000

Alternativa interesante:

- 2D FFT batched para datos tipo imagen

### Metricas recomendadas

No recomiendo expresar FFT en `GFLOP/s` como metrica principal, porque el conteo de FLOPs en FFT depende de convenciones y puede inducir debate innecesario.

Recomiendo usar:

- `time_ms`
- `transforms_per_second`
- `MSamples/s` o `GSamples/s`
- opcionalmente `effective GB/s`

Para energia:

- `transforms/s/W`
- `MSamples/s/W`


## Diseño experimental recomendado para FFT

### Objetivo

Que sea un benchmark:

- reproducible
- no demasiado pequeno
- medible energeticamente
- representativo

### Regla importante

No volver a cometer el mismo error de los casos cortos.

Por tanto:

- el caso elegido para energia debe durar claramente mas de `10 ms`
- idealmente, mucho mas de `100 ms`
- mejor aun si puede mantenerse del orden de `1 s`

### Estrategia recomendada

Separar:

- `FFT baseline sweep`
- `FFT energy case`

Donde:

- el baseline explora varios tamanos y batches
- el energy case elige uno o dos casos largos y estables

Esto seria mas robusto que medir energia sobre un caso "peak" ultracorto.


## Si se añade tambien CUB

## Opcion secundaria muy valiosa - `sort_bench.cu` o `reduce_bench.cu`

Implementacion posible:

- `sort_bench.cu` con `cub::DeviceRadixSort`
- o `reduce_bench.cu` con `cub::DeviceReduce`

Metricas:

- `keys/s`
- `GB/s efectivos`
- `items/s/W`

Valor añadido:

- benchmark claramente no matricial
- muy bueno para control de throughput/memoria
- probablemente mas equilibrado entre GPUs que GEMM TF32

Si el tiempo es limitado:

- priorizar primero `cuFFT`
- dejar `CUB` como trabajo futuro o mejora adicional


## Como contarlo en la memoria

Esta tarea se puede justificar asi:

- los benchmarks iniciales cubren ancho de banda, FMA y GEMM
- eso da una vision util, pero parcialmente sesgada hacia algebra lineal densa
- se añade un benchmark no matricial basado en biblioteca oficial para ampliar la validez externa del estudio
- el nuevo benchmark actua como carga de control y permite comparar otra familia de workloads CUDA


## Bibliografia y documentacion a consultar

Fuentes oficiales prioritarias:

- CUDA docs portal:
  - https://docs.nvidia.com/cuda/doc/
- cuFFT:
  - https://docs.nvidia.com/cuda/archive/12.6.3/cufft/index.html
  - https://docs.nvidia.com/cuda/archive/12.6.1/cufft/contents.html
- CUB / CCCL:
  - https://nvidia.github.io/cccl/cub/
  - https://nvidia.github.io/cccl/cub/device_wide.html
  - https://nvidia.github.io/cccl/cub/api/file_cub_device_device_radix_sort.cuh.html
  - https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceReduce.html
- NPP:
  - https://docs.nvidia.com/cuda/npp/introduction.html
- cuRAND:
  - https://docs.nvidia.com/cuda/curand/index.html

Temas de busqueda bibliografica:

- FFT en GPU
- rendimiento de FFT frente a GEMM en arquitecturas NVIDIA
- transformadas batched en GPU
- benchmarks de primitivas paralelas en GPU
- radix sort GPU benchmark
- throughput y energy efficiency en cargas memory-bound vs compute-bound


## Recomendacion ejecutiva final

Si hubiera que tomar una decision hoy:

- implementar `cuFFT` como nuevo benchmark no matricial principal
- disenar los casos de energia para que sean largos y estables
- usar `CUB` como opcion secundaria si hay tiempo

Esto cumple mejor la peticion del profesor porque:

- amplia el estudio mas alla de GEMM
- evita centrar toda la narrativa en una comparacion que favorece demasiado a la A100 por arquitectura
- sigue usando tecnologia fiable, oficial y defendible academicamente
