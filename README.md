# Trabajo 1 de Redes Neuronales Artificiales y Algoritmos Bio-Inspirados de la UNAL-med
# Optimización Eurística

## 1. Introducción
En el mundo de las redes neuronales y los algoritmos bioinspirados existen, al igual que en la naturaleza, muchos medios de alcanzar un objetivo (en este caso, minimizar una función; en la naturaleza: comer, reproducirse, adaptarse, sobrevivir). Por lo mismo, se trabajará utilizando diferentes tipos de optimizaciones para analizar qué ventajas y diferencias tienen entre ellas. Las que se utilizarán son: Método de descenso por gradiente con condición inicial aleatoria, algoritmos evolutivos, optimización de partículas y evolución diferencial. Además, se desea observar qué diferencias resultan de usar métodos de uso de lógica matemática, como descenso por gradiente a diferencia de los que puedan obtenerse de los bioinspirados. 

Para el análisis de las optimizaciones, se usarán dos funciones: La función de las seis jorobas de camello [1] y la función Rosenbrock [2].

### Seis Jorobas de Camello:
Es una función de dos variables $f(x_1,x_2) = y$. 

$(f(x) = 2*x_1^2 - 1.05*x_1^4 + x_1^6 /6 + x_1*x_2 + x_2^2)$

Por lo general se evalua en $x_i \in [-5,5],\ i = 1,2$

Puede ser engañoso dado que tiene múltiples mínimos locales. Su mínimo global es $(x_1, x_2) = (0, 0) $

#### FOTO

### Rosenbrock:
Es una función para n variables en $f(x_1,x_2,...,x_n) = y $

$f(x) = \sum([100*(x_{i+1}-x_i^2)^2 + (x_i-1)^2])$

## A cargo de:

- Esteban García Carmona
- Emilio Porras Mejía
- Felipe Miranda Arboleda

## Introducción

## 1. Optimización Numérica

## 2. Optimización Combinatoria

## Problemática

Se debe resolver el problema del vendedor viajero para varias ciudades principales de Colombia. El objetivo es hallar un camino que, pasando por cada ciudad por lo menos una vez, minimice el costo definido por la suma del valor de la hora del vendedor, el costo de los peajes y el costo del combustible.

## Procedimiento

Se solucionó el problema por dos vías diferentes: Algoritmos genéticos (GA) y Colonia de hormigas (AC).

1. Creamos la base de datos de las ciudades a visitar con sus respectivas coordenadas [1].

| Cod Mun | Municipio    | Cod Dep | Departamento       | Latitud     | Longitud     |
|---------|--------------|---------|--------------------|-------------|--------------|
| 63001   | Armenia      | 63      | Quindio            | 4,5338889   | -75,6811111  |
| 8001    | Barranquilla | 8       | Atlantico          | 10,9638889  | -74,7963889  |
| 11001   | Bogota D.C.  | 25      | Cundinamarca       | 4,6         | -74,0833333  |
| 68001   | Bucaramanga  | 68      | Santander          | 7,1297222   | -73,1258333  |
| 13001   | Cartagena    | 13      | Bolivar            | 10,3997222  | -75,5144444  |
| 54001   | Cucuta       | 54      | Norte de Santander | 7,8833333   | -72,5052778  |
| 17001   | Manizales    | 17      | Caldas             | 5,07        | -75,5205556  |
| 5001    | Medellin     | 5       | Antioquia          | 6,2913889   | -75,5361111  |
| 23001   | Monteria     | 23      | Cordoba            | 8,7575      | -75,89       |
| 76520   | Palmira      | 76      | Valle del Cauca    | 3,5394444   | -76,3036111  |
| 66001   | Pereira      | 66      | Risaralda          | 4,8133333   | -75,6961111  |
| 52001   | Pasto        | 52      | Narino             | 1,214670737 | -77,27864742 |
| 8758    | Soledad      | 8       | Atlantico          | 10,9172222  | -74,7666667  |
| 76834   | Tulua        | 76      | Valle del Cauca    | 4,0866667   | -76,2        |
| 20001   | Valledupar   | 20      | Cesar              | 10,4769444  | -73,2505556  |

2. Procedemos a construír la matriz de costos. La entrada i,j de esta matriz contendrá el costo en pesos de ir de la ciudad i a la ciudad j. Éste, a su vez es la suma del salario por horas del vendedor, costo de los peajes y el costo del combustible en valores del 2023.

    Costo hora del vendedor: El salario promedio de un conductor en colombia es de $6.827/hora [2].

    Costo de los peajes: Se estableció el costo de los peajes, el tiempo estimado de viaje y la distancia para cada par de ciudades [3].

    Costo del combustible: El recorrido se hará en un Mini Cooper 1.6, cuyo rendimiento es de 8,35 litros / 100 km [4]. Además, el costo promedio de un litro de gasolina corriente en colombia es de $2.747,31/litro [5]. 

    Importamos las tablas de tiempo de viaje, costo de los peajes y distancia entre ciudades. Para así calcular la siguiente matriz de costos totales en pesos:

| Armenia | Barranquilla | Bogota D.C. | Bucaramanga | Cartagena | Cucuta | Manizales | Medellin | Monteria | Palmira | Pereira | Pasto | Soledad | Tulua | Valledupar |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 484086.92267999996 | 158390.37857 | 276519.39869 | 488048.897655 | 373928.47954 | 75171.62696 | 154821.05318 | 386649.35525499994 | 109939.39929999999 | 33138.218479999996 | 272342.285215 | 484086.92267999996 | 109939.39929999999 | 418535.755365 |
| 484086.92267999996 | 0.0 | 489627.626915 | 284902.81522499997 | 74886.8562 | 302841.99294499995 | 436731.38418999995 | 355567.88026999997 | 163759.57590499998 | 610085.74202 | 493162.99425 | 773854.0279349999 | 0.0 | 610085.74202 | 155067.773575 |
| 158390.37857 | 489627.626915 | 0.0 | 200351.15669499998 | 502543.99425 | 248886.51444499998 | 154480.32627 | 232200.960555 | 434579.62032 | 270600.13864 | 185157.48242999997 | 431240.645325 | 489627.626915 | 270600.13864 | 390112.273795 |
| 276519.39869 | 284902.81522499997 | 200351.15669499998 | 0.0 | 306205.39025 | 94314.92623 | 227708.18558 | 218759.15553999998 | 331638.676045 | 399633.056065 | 271317.30521499994 | 551325.6389 | 284902.81522499997 | 399633.056065 | 201113.822865 |
| 488048.897655 | 74886.8562 | 502543.99425 | 306205.39025 | 0.0 | 315763.86143499997 | 431280.009935 | 334374.8464 | 152632.20626 | 577374.299255 | 452510.226865 | 735962.15209 | 74886.8562 | 577374.299255 | 168573.980525 |
| 373928.47954 | 302841.99294499995 | 248886.51444499998 | 94314.92623 | 315763.86143499997 | 0.0 | 317427.01719 | 300072.69022999995 | 348359.996075 | 478131.01729 | 361290.12759499997 | 639821.111665 | 302841.99294499995 | 478131.01729 | 215874.40212499996 |
| 75171.62696 | 436731.38418999995 | 154480.32627 | 227708.18558 | 431280.009935 | 317427.01719 | 0.0 | 99300.11468999999 | 322660.35099999997 | 161951.94933 | 44775.21079 | 252075.94324 | 436731.38418999995 | 161951.94933 | 366919.97109999997 |
| 154821.05318 | 355567.88026999997 | 232200.960555 | 218759.15553999998 | 334374.8464 | 300072.69022999995 | 99300.11468999999 | 0.0 | 234263.495925 | 244004.413625 | 119375.24277499999 | 405334.06877 | 355567.88026999997 | 244004.413625 | 347255.136825 |
| 386649.35525499994 | 163759.57590499998 | 434579.62032 | 331638.676045 | 152632.20626 | 348359.996075 | 322660.35099999997 | 234263.495925 | 0.0 | 458683.35762499995 | 344100.87908499996 | 625157.8531549999 | 163759.57590499998 | 458683.35762499995 | 200927.126705 |
| 109939.39929999999 | 610085.74202 | 270600.13864 | 399633.056065 | 577374.299255 | 478131.01729 | 161951.94933 | 244004.413625 | 458683.35762499995 | 0.0 | 127769.59085 | 177239.63937999998 | 610085.74202 | 0.0 | 525409.23966 |
| 33138.218479999996 | 493162.99425 | 185157.48242999997 | 271317.30521499994 | 452510.226865 | 361290.12759499997 | 44775.21079 | 119375.24277499999 | 344100.87908499996 | 127769.59085 | 0.0 | 278667.52407 | 493162.99425 | 127769.59085 | 409963.84150499996 |
| 272342.285215 | 773854.0279349999 | 431240.645325 | 551325.6389 | 735962.15209 | 639821.111665 | 252075.94324 | 405334.06877 | 625157.8531549999 | 177239.63937999998 | 278667.52407 | 0.0 | 773854.0279349999 | 177239.63937999998 | 685441.7648049999 |
| 32102800.788 | 0.0 | 489627.626915 | 284902.81522499997 | 74886.8562 | 302841.99294499995 | 436731.38418999995 | 355567.88026999997 | 163759.57590499998 | 610085.74202 | 493162.99425 | 773854.0279349999 | 0.0 | 610085.74202 | 155067.773575 |
| 11446906.425999999 | 44665970.880499996 | 270600.13864 | 399633.056065 | 577374.299255 | 478131.01729 | 161951.94933 | 244004.413625 | 458683.35762499995 | 0.0 | 127769.59085 | 177239.63937999998 | 610085.74202 | 0.0 | 525409.23966 |
| 418535.755365 | 155067.773575 | 390112.273795 | 201113.822865 | 168573.980525 | 215874.40212499996 | 366919.97109999997 | 347255.136825 | 200927.126705 | 525409.23966 | 409963.84150499996 | 685441.7648049999 | 155067.773575 | 525409.23966 | 0.0 |

3. Se implementaron algoritmos genéticos por medio de python [6] y se obtuvo el siguiente gif que representa la evolución de la mejor ruta a lo largo de las generaciones:

    <img src="gifGA.gif" alt="mapaga" title="mapaga">

    _figura x: Mapa de Colombia con recorridos_ 

    Con lo que la mejor ruta encontrada por los algoritmos genéticos fue:

    Pasto -> Palmira -> Tulua -> Armenia -> Pereira -> Medellin -> Manizales -> Bogota D.C. -> Bucaramanga -> Cucuta -> Valledupar -> Soledad -> Barranquilla -> Cartagena -> Monteria

    Con un costo de: $1.586.600 COP

4. Se implementó colonia de hormigas por medio de python [7] y se obtuvo el siguiente gif que representa la evolución de la mejor ruta que descubren 100 hormigas:

    <img src="gifAC.gif" alt="mapaga" title="mapaga">

    _figura x: Mapa de Colombia con recorridos de hormigas_ 

    Con lo que la mejor ruta encontrada por la colonia de hormigas fue:

    Pereira -> Armenia -> Palmira -> Tulua -> Manizales -> Pasto -> Bogota D.C. -> Bucaramanga -> Cucuta -> Valledupar -> Soledad -> Barranquilla -> Cartagena -> Monteria -> Medellin

    Con un costo de: $2.013.640 COP

## Conclusiones

- Ambos algoritmos lograron solucionar el problema con resultados mucho mejores que el azar o un análisis a priori.

- Los algoritmos genéticos se desempeñaron mejor que las colonias de hormigas, pero se sospecha que se debe a la capacidad de cómputo limitada que solo permitió simular 100 hormigas.

- Recorrer todas las ciudades deseadas por poco más de un millón y medio de pesos parece muy razonable, más aún teniendo en cuenta que se escogió un vehículo que no tiene un rendimiento de combustible destacable.



## Bibliografía y referencias
[1] "Virtual Library of Simulation Experiments" (2013). Three-Hump Camel Function [Online]. Available: https://www.sfu.ca/~ssurjano/camel3.html
[2] "Virtual Library of Simulation Experiments" (2013). Rosenbrock Function [Online]. Available: https://www.sfu.ca/~ssurjano/rosen.html

- [1] “Geoportal del DANE - Codificación Divipola,” geoportal.dane.gov.co. https://geoportal.dane.gov.co/geovisores/territorio/consulta-divipola-division-politico-administrativa-de-colombia/ (accessed Mar. 10, 2023).

- [2] “Salario para Conductor en Colombia - Salario Medio,” Talent.com. https://co.talent.com/salary?job=conductor#:~:text= (accessed Mar. 10, 2023).

- [3] S. O. C. S.A.S, “Peajes en Colombia [2022],” Viaja por Colombia. https://viajaporcolombia.com/peajes/ (accessed Mar. 10, 2023).

- [4] “Consumo Gasolina: 8,35 l/100km - Mini, Mini Cooper, Mini Cooper 1.6,” www.spritmonitor.de. https://www.spritmonitor.de/es/detalle/125236.html?cdetail=1 (accessed Mar. 10, 2023).

- [5] “Colombia precios de la gasolina, 06-marzo-2023,” GlobalPetrolPrices.com. https://es.globalpetrolprices.com/Colombia/gasoline_prices/#:~:text=El%20valor%20medio%20durante%20este (accessed Mar. 10, 2023).

- [6] M. Kukreja, “Travelling-Salesman-Problem-with-Genetic-Algorithm,” GitHub, Oct. 10, 2022. https://github.com/manpreet1130/Travelling-Salesman-Problem-with-Genetic-Algorithm (accessed Mar. 10, 2023).
‌
- [7] R. Zhang, “ant-colony-tsp,” GitHub, Feb. 27, 2023. https://github.com/ppoffice/ant-colony-tsp (accessed Mar. 10, 2023).