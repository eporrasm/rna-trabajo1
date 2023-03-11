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





## Bibliografía y referencias
[1] "Virtual Library of Simulation Experiments" (2013). Three-Hump Camel Function [Online]. Available: https://www.sfu.ca/~ssurjano/camel3.html
[2] "Virtual Library of Simulation Experiments" (2013). Rosenbrock Function [Online]. Available: https://www.sfu.ca/~ssurjano/rosen.html


Coordenadas de municipios: https://geoportal.dane.gov.co/geovisores/territorio/consulta-divipola-division-politico-administrativa-de-colombia/
