# Protocolo incremental de E1

## Pregunta causal

La unidad de evidencia no es que baje el loss, sino que una representacion de
spikes generalice relaciones semanticas y que conservar el tiempo aporte mas
que conservar solamente el conteo. Se mantienen separadas estas preguntas:

1. ¿La arquitectura puede optimizar una senal semantica?
2. ¿Generaliza fuera de las asociaciones supervisadas?
3. ¿La informacion util esta en el timing y no solo en firing rates?
4. ¿La dinamica spiking ofrece algo frente a una ANN temporal comparable?

## Iteracion 0 — smoke test (implementada)

- Entrada: un evento one-hot por grafema, intervalo fijo y silencio final.
- Modelo: tres poblaciones ALIF recurrentes, taus heterogeneas/aprendibles.
- Salida: spike train binario de la ultima poblacion.
- Objetivo: triplet loss sobre Van Rossum multiescala mas control de actividad.
- Controles inmediatos: SNN sin entrenar y distancia rate-only.
- Alcance: vocabulario pequeno e IID. Sirve para G0/G1, no para H3/H5.

## Iteracion 1 — controles temporales (implementada)

Sin aumentar parametros:

- fijar un conjunto de evaluacion sin duplicados y balanceado por tipo;
- ejecutar 5 semillas y reportar media e IC bootstrap;
- agregar shuffle local de radios 1 y 3 y shuffle global, preservando
  exactamente el conteo de cada neurona;
- medir STA por positivos semanticos y negativos ortograficos dificiles;
- guardar actividad por capa y separar ventana de entrada/post-estimulo;
- comparar taus multiples contra una sola tau sin cambiar el resto.

### Resultado E1.1

Se entrenaron cinco seeds durante 30 epocas sobre CPU. La validacion fija tiene
260 tripletas: 240 con negativos aleatorios y 20 con negativos ortograficos
dificiles. Cada shuffle se repitio cinco veces por seed.

| Control | STA media | IC bootstrap 95% |
|---|---:|---:|
| Van Rossum original | 0.910 | [0.876, 0.935] |
| Shuffle local, radio 1 | 0.910 | [0.876, 0.935] |
| Shuffle local, radio 3 | 0.913 | [0.879, 0.941] |
| Shuffle global | 0.886 | [0.848, 0.914] |
| Rate-only | 0.913 | [0.866, 0.945] |

La diferencia pareada original menos global fue `+0.024` [0.009, 0.039], pero
original menos rate-only fue `-0.003` [-0.015, 0.012]. El shuffle global agrega
ruido a Van Rossum, mientras rate-only conserva toda la informacion necesaria.
Por eso E1.1 **no respalda H3** en esta configuracion. Tampoco hay sensibilidad
local: corromper uno o tres pasos no reduce el resultado.

Por dificultad, Van Rossum original obtuvo 0.932 en negativos aleatorios y
0.640 en negativos ortograficos. Rate-only obtuvo respectivamente 0.929 y
0.720. La debilidad principal esta en separar formas superficialmente parecidas.
La distancia media de los positivos semanticos fue 0.608, frente a 1.458 para
negativos aleatorios y 0.763 para negativos ortograficos; estos ultimos quedan
demasiado cerca de los positivos y explican su menor STA.

La STA por capa fue 0.632 -> 0.768 -> 0.910. Esto es un indicio de organizacion
jerarquica, pero aun no es el indice de convergencia positivo/negativo requerido
para aceptar H2. La actividad post-estimulo crecio por capa: 0.001, 0.021 y
0.045, indicando que las regiones tardias mantienen dinamica durante el silencio.

Estado despues de E1.1:

- G0 — Pipeline funcional: aprobado.
- G1 — Micro-aprendizaje: aprobado.
- G2 — Reproducibilidad: aprobado para este conjunto IID pequeno.
- H1 — Estructura semantica: indicio IID; no demostrada fuera del vocabulario.
- H2 — Convergencia por capas: indicio; falta medir convergencia directamente.
- H3 — Temporal coding: no respaldada por E1.1.
- H4 — Ventaja multiescala: pendiente.
- H5 — Generalizacion: pendiente.
- Comparacion ANN: pendiente.

El archivo autoritativo con resultados por seed, curvas por dificultad,
actividad por capa e intervalos pareados es `runs/e11/summary.json`.

## Iteracion 2 — dataset y generalizacion

El dataset versionado tendra registros con `anchor`, `positive`, `negative`,
`relation`, `source`, `lexical_family` y `split`. Un auditor rechazara familias
solapadas. Se reportaran tres escenarios distintos:

- IID: mismas familias y vocabulario;
- morphology-disjoint: nuevas formas de familias conocidas;
- lexical-disjoint contextual: lexemas nuevos que solo reciben evidencia en
  contextos no supervisados/informativos.

Una prueba con palabras aisladas nunca vistas no se considerara una medicion
valida de semantica, porque la entrada no contiene evidencia identificable.

## E1.2A — sensibilidad de la metrica y margen por capa (implementada)

Se evaluaron los checkpoints congelados de E1.1, sin reentrenamiento. Cada
subconjunto conserva los pesos originales de las escalas incluidas y los
renormaliza.

| Metrica | STA total | STA random | STA ortografica dificil |
|---|---:|---:|---:|
| tau 2 | 0.872 | 0.901 | 0.520 |
| tau 6 | 0.902 | 0.927 | 0.590 |
| tau 14 | 0.914 | 0.935 | 0.660 |
| tau 2+6 | 0.896 | 0.923 | 0.580 |
| tau 2+14 | 0.912 | 0.934 | 0.650 |
| tau 6+14 | 0.911 | 0.933 | 0.640 |
| tau 2+6+14 | 0.910 | 0.932 | 0.640 |
| rate-only | 0.913 | 0.929 | 0.720 |

`tau=14` supera a la combinacion completa por 0.004 [0.001, 0.008] y a
`tau=2` por 0.042 [0.029, 0.055]. Frente a rate-only la diferencia es 0.001
[-0.012, 0.016]. Por tanto, H4a no recibe apoyo: combinar escalas no mejora la
metrica; la escala lenta sola es suficiente y la escala rapida degrada
especialmente los distractores ortograficos. H4b sigue siendo una pregunta de
reentrenamiento distinta.

### H2 — separacion jerarquica

Con Van Rossum multiescala original:

| Subconjunto/capa | D positivo | D negativo | Margen | Margen relativo | STA |
|---|---:|---:|---:|---:|---:|
| Todos L1 | 0.091 | 0.100 | 0.009 | 0.172 | 0.632 |
| Todos L2 | 0.289 | 0.397 | 0.108 | 0.489 | 0.768 |
| Todos L3 | 0.608 | 1.404 | 0.797 | 1.637 | 0.910 |
| Random L1 | 0.091 | 0.106 | 0.014 | 0.229 | 0.680 |
| Random L2 | 0.289 | 0.414 | 0.125 | 0.552 | 0.815 |
| Random L3 | 0.608 | 1.458 | 0.850 | 1.734 | 0.932 |
| Ortografico L1 | 0.089 | 0.040 | -0.049 | -0.516 | 0.050 |
| Ortografico L2 | 0.289 | 0.191 | -0.098 | -0.261 | 0.210 |
| Ortografico L3 | 0.607 | 0.763 | 0.156 | 0.476 | 0.640 |

Las distancias absolutas crecen con la actividad, de modo que no hay
"convergencia" literal de positivos. Si se formula H2 como separacion
jerarquica, la evidencia es fuerte: el margen relativo total crece en cada
capa. Para distractores ortograficos, las capas 1 y 2 los consideran mas
cercanos que los positivos semanticos; solo la capa 3 invierte el orden y
produce margen positivo. Esto observa directamente una transformacion desde
similitud superficial hacia separacion semantica, aunque aun incompleta.

La siguiente prueba factorial de forma sera una matriz 2x2: ortografia
similar/distinta por semantica similar/distinta, medida en cada capa.

Estado despues de E1.2A:

- G0/G1/G2: aprobados en el regimen IID actual.
- H1 — Semantica IID: evidencia inicial; limitada por ortografia.
- H2 — Jerarquia por profundidad: evidencia positiva como separacion, no como
  convergencia absoluta.
- H3 — Timing semantico: no respaldada en E1.1.
- H4a — Ventaja de la metrica multiescala: no respaldada; tau 14 funciona mejor.
- H4b — Dinamica neuronal multiescala: pendiente de reentrenamiento controlado.
- H5, matriz factorial ortografica y baseline ANN: pendientes.

## E1.2C — orden y timing de entrada (implementada)

Se evaluaron los mismos checkpoints congelados. Todas las corrupciones conservan
exactamente el numero de eventos de cada canal de grafema. `reverse` y
`permuted_order` conservan los tiempos regulares; los jitters conservan orden y
el primer evento en `t=0`, variando solo los intervalos. Cada condicion aleatoria
se repitio cinco veces por seed.

STA de la capa conceptual:

| Entrada | Van Rossum | Rate-only | Margen VR |
|---|---:|---:|---:|
| Correcta | 0.910 | 0.913 | 0.797 |
| Reverse | 0.657 | 0.630 | 0.265 |
| Orden permutado | 0.645 | 0.654 | 0.250 |
| Orden correcto, gaps 1–3 | 0.862 | 0.867 | 0.676 |
| Orden correcto, gaps 1–5 | 0.808 | 0.824 | 0.623 |
| Bag-of-graphemes | 0.588 | 0.640 | 0.154 |

Caidas pareadas de rate-only frente a entrada correcta:

- reverse: 0.283 [0.246, 0.314];
- permutacion: 0.259 [0.197, 0.295];
- jitter gaps 1–3: 0.046 [0.015, 0.074];
- jitter gaps 1–5: 0.089 [0.046, 0.126];
- bag-of-graphemes: 0.273 [0.210, 0.311].

En negativos ortograficos, rate-only cae de 0.720 a 0.422 con permutacion y a
0.170 con reverse. La transformacion semantica de la capa final depende
fuertemente del orden de entrada. El jitter muestra una degradacion gradual al
alterar el ritmo sin alterar el orden.

Esto respalda **H3a (el procesamiento ordenado/secuencial es necesario para
construir el estado)** bajo el regimen entrenado. Es compatible con que la
dinamica use la historia para construir un codigo final predominantemente
poblacional. No demuestra que intervalos precisos sean intrinsecamente
informativos.
No demuestra precise output spike timing, causalidad especifica de una SNN ni
robustez a ritmos no vistos: las corrupciones son fuera de distribucion y una
ANN secuencial podria mostrar el mismo patron. La sensibilidad al jitter tambien
indica que futuras entradas de audio necesitaran augmentation temporal o una
representacion de timing mas natural.

Estado despues de E1.2C:

- H2 — Jerarquia por profundidad: evidencia positiva.
- H3a — Orden/tiempo de entrada para construir el estado: respaldada.
- H3b — Timing preciso del codigo final: no respaldada.
- H3c — Invariancia controlada a tempo conservando orden: pendiente; el jitter
  actual es OOD porque el entrenamiento utilizo intervalos fijos.
- H4a — Ventaja de la metrica multiescala: no respaldada.
- H4b — Dinamica neuronal multiescala: siguiente reentrenamiento controlado.

## E1.2B — escalas temporales neuronales (implementada)

Esta ablacion modifica exclusivamente el decay de membrana `beta`. Recurrencia,
adaptacion, decay adaptativo, pesos, datos, loss, regularizacion, seeds y epocas
permanecen iguales. Las variantes fijas tienen 17,968 parametros entrenables y
la aprendible 18,128; todas poseen 18,128 valores de estado. La diferencia de
160 parametros es menor al 1%.

| Dinamica de membrana | STA total | STA hard | Margen hard L3 | Post-rate L3 |
|---|---:|---:|---:|---:|
| Un beta fijo global | 0.892 | 0.490 | 0.029 | 0.049 |
| Beta fijo por capa | 0.900 | 0.520 | 0.086 | 0.055 |
| Heterogeneo fijo | 0.885 | 0.520 | 0.126 | 0.036 |
| Heterogeneo aprendible | **0.910** | **0.640** | **0.156** | 0.045 |

Comparaciones pareadas de heterogeneo aprendible:

- frente a beta unico: STA total +0.018 [-0.032, 0.065], STA hard +0.150
  [0.020, 0.280], margen hard +0.127 [0.016, 0.226];
- frente a beta por capa: STA hard +0.120 [-0.050, 0.300], margen +0.070
  [-0.127, 0.269];
- frente a heterogeneo fijo: STA hard +0.120 [-0.010, 0.250], margen +0.030
  [-0.124, 0.185].

No hay diferencia consistente de actividad post-estimulo. Los `beta` aprendidos
tampoco se alejan drasticamente de su inicializacion: sus medias finales por
capa son 0.557, 0.682 y 0.809. La mejora no puede atribuirse a un cambio global
grande de constante temporal; podria depender de ajustes neuronales pequenos.

H4b recibe **evidencia parcial**. La dinamica aprendible mejora de manera
reproducible el caso ortografico frente a una unica constante, precisamente
donde debe ocurrir la abstraccion forma-semantica. Con cinco seeds no se puede
separar concluyentemente el aporte de "por capa", "heterogeneidad" y
"aprendibilidad". La STA global tampoco muestra ventaja concluyente.

Progresion hard por capas:

| Variante | STA L1 | STA L2 | STA L3 | Margen L3 |
|---|---:|---:|---:|---:|
| Un beta | 0.070 | 0.270 | 0.490 | 0.029 |
| Por capa | 0.040 | 0.140 | 0.520 | 0.086 |
| Heterogeneo fijo | 0.060 | 0.270 | 0.520 | 0.126 |
| Heterogeneo aprendible | 0.050 | 0.210 | **0.640** | **0.156** |

Estado despues de E1.2B:

- H3a — Dependencia secuencial: respaldada.
- H3b — Timing preciso de salida: no respaldada.
- H3c — Invariancia de tempo: pendiente para E1.3.
- H4a — Metrica multiescala: no respaldada.
- H4b — Dinamica neuronal multiescala: evidencia parcial, localizada en
  distractores ortograficos.
- Siguiente: matriz factorial 2x2 y despues E1.3 con augmentation temporal.

### Diagnostico de especializacion de beta

La inicializacion aprendible ya era heterogenea dentro de cada capa. El analisis
por neurona no encontro una diversificacion oculta fuerte en L3:

| Capa | std inicial | std final | media abs(delta beta) |
|---|---:|---:|---:|
| L1 | 0.0472 | 0.0510 | 0.0127 |
| L2 | 0.0469 | 0.0466 | 0.0077 |
| L3 | 0.0472 | 0.0469 | 0.0055 |

En L3, las correlaciones de beta o `abs(delta beta)` con firing dificil y
actividad post-estimulo son compatibles con cero. L2 muestra asociaciones
pequenas: Spearman entre `abs(delta beta)` y firing general 0.145 [0.057, 0.234],
y con firing dificil 0.148 [0.044, 0.252].

No hay evidencia de una especializacion temporal fuerte creada por aprendizaje
en la capa conceptual. La explicacion mas parsimoniosa es que la inicializacion
heterogenea/por capas ya proporciona la estructura util; ajustes pequenos pueden
contribuir, pero no explican por si solos H4b. Esto coincide con que aprendible
no supera concluyentemente a heterogeneo fijo.

## Matriz factorial 2x2 — piloto diagnostico (implementada)

Se construyeron 48 pares curados, 12 por celda. La similitud Levenshtein media
es 0.795 en las celdas `orth+` y 0.117 en `orth-`. La celda semantica positiva
usa inflexion cuando `orth+` y sinonimia/parafrasis cuando `orth-`.

Distancia Van Rossum media:

| Celda | L1 | L2 | L3 |
|---|---:|---:|---:|
| orth+ / semantic+ | 0.013 | 0.064 | 0.187 |
| orth+ / semantic- | 0.039 | 0.178 | 0.661 |
| orth- / semantic+ | 0.098 | 0.319 | 0.627 |
| orth- / semantic- | 0.108 | 0.434 | 1.611 |

Efectos relativos y `A = efecto semantico - efecto ortografico`:

| Readout | Capa | Semantica S | Ortografia O | Abstraccion A |
|---|---:|---:|---:|---:|
| Van Rossum | L1 | 0.278 | 1.188 | -0.911 |
| Van Rossum | L2 | 0.472 | 1.044 | -0.571 |
| Van Rossum | L3 | 0.940 | 0.905 | +0.035 |
| Rate-only | L1 | 0.330 | 1.184 | -0.854 |
| Rate-only | L2 | 0.669 | 1.082 | -0.413 |
| Rate-only | L3 | 1.172 | 0.984 | +0.188 |

El IC de `A` Van Rossum L3 es [-0.084, 0.171]; llega a equilibrio pero no es
concluyente. En rate-only L3, `A=0.188` [0.075, 0.304], de modo que la semantica
supera a la ortografia dentro de este conjunto fijo. La inversion aparece solo
en la capa final y refuerza la lectura poblacional de H2.

Este resultado sigue siendo piloto, no una prueba causal factorial: la
exposicion supervisada difiere entre celdas, frecuencia lexica no esta
controlada, las relaciones semanticas orth+ y orth- no son del mismo tipo y el
bootstrap entre seeds no representa incertidumbre sobre la seleccion de pares.
La conclusion permitida es que los checkpoints muestran el patron interno
forma->semantica sobre un conjunto curado; no que generalicen ese efecto.

Estado despues de la matriz:

- H2 — Inversion forma-semantica por profundidad: respaldada como diagnostico
  IID/poblacional; falta replicacion lexicalmente independiente.
- H4b — La multiescala ayuda frente a beta unico: evidencia parcial; no hay
  evidencia suficiente de que aprender beta sea el componente decisivo.
- Siguiente: E1.3/H3c, tempo augmentation con test fuera del rango visto.

## Baseline ANN pareado (implementado)

La ANN continua tiene los mismos 18,128 parametros, entrada, profundidades,
anchos, recurrencia, adaptacion, beta, seeds, datos y loss contrastivo. Sustituye
el disparo binario por estados adaptive leaky-tanh.

| Metrica poblacional/integrada | SNN | ANN | Diferencia SNN-ANN |
|---|---:|---:|---:|
| STA IID | 0.913 | 0.928 | -0.015 [-0.056, 0.012] |
| STA ortografica dificil | 0.720 | 0.540 | +0.180 [0.040, 0.360] |
| Indice factorial A3 | 0.188 | 0.141 | +0.046 [-0.103, 0.142] |
| Caida por permutar entrada | 0.259 | 0.133 | +0.127 [0.059, 0.166] |

La inversion forma-semantica tambien aparece en ANN, por lo que no es especifica
de spikes. La ANN es ligeramente mejor globalmente sin diferencia concluyente;
la SNN separa mejor distractores ortograficos y depende mas del orden. La tasa
de spikes SNN es 0.090, mientras 0.986 de los estados ANN superan `abs(1e-3)`;
esto mide sparsity operacional, no energia.

## Generalizacion (implementada)

### Relation-disjoint

Ambos lexemas aparecen en entrenamiento, pero la arista positiva evaluada y su
inversa se excluyen. STA poblacional: SNN 0.778 [0.706, 0.844]; ANN 0.906
[0.850, 0.961].

### Transferencia lexica con exposicion contextual

Ocho lexemas nuevos aparecen solo dentro de tripletas de contextos semanticamente
supervisadas y despues se evalúan como cadenas aisladas. El control sin contexto
recibe el mismo numero de ejemplos y actualizaciones.

| Modelo | Sin contexto | Con contexto | Ganancia pareada |
|---|---:|---:|---:|
| SNN | 0.487 | 0.758 | +0.271 [0.208, 0.333] |
| ANN | 0.546 | 0.833 | +0.288 [0.254, 0.313] |

Existe transferencia contextual hacia palabras aisladas en ambos modelos. La
ANN generaliza mejor: +0.128 en relation-disjoint y +0.075 en transferencia
contextual, con intervalos pareados que no cruzan cero. Este protocolo no es
exposicion no supervisada: la señal semantica actua sobre oraciones que contienen
el nuevo lexema.

Estado de cierre del nucleo E1:

- transformacion jerarquica forma-semantica: respaldada en regimen pequeno;
- especificidad SNN de esa transformacion: no respaldada;
- ventaja SNN: hard negatives, sparsity y mayor dependencia secuencial;
- ventaja ANN: generalizacion y rendimiento global ligeramente mayor;
- generalizacion relacional/contextual: respaldada de forma estrecha;
- tempo invariance, dataset externo y adquisicion no supervisada: abiertos.

## Iteracion 3 — arquitectura y baselines

Solo despues de superar las puertas anteriores se incorporan delays/dilations
explicitos, trazas por capa, convergencia y 2–10 M de parametros. En ese punto
se comparan SNN aleatoria, ANN temporal pareada, SNN de una escala y E1 completa.

## Resultado registrado de la primera corrida

Configuracion: semilla 7, 18,128 parametros, CPU, 30 epocas, 76 tripletas de
entrenamiento y 20 de validacion. El archivo autoritativo es
`runs/smoke/summary.json` (generado, no versionado).

- G1: aprobado en la corrida observada.
- STA temporal IID: 0.50 inicial -> 0.95 final.
- STA rate-only final: 0.95.
- Spike rate final: 0.081.
- Neuronas activas finales: 0.958.
- Interpretacion: aprende sin colapso; no existe aun evidencia incremental de
  temporal coding, generalizacion lexical-disjoint ni ventaja frente a ANN.
