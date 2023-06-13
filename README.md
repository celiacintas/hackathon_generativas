# Introducción a modelos generativos de aprendizaje profundo robustos
### Docentes: Dra. Celia Cintas (IBM Research), Dr. Pablo Navarro (CONICET) 

Los modelos generativos se ajustan a una distribución conocida de datos, representándolos como un espacio latente. Los recientes avances en el entrenamiento de redes profundas como modelos generativos han resultado en un gran crecimiento de la investigación de estos modelos, como evaluarlos, filtrado y sus potenciales aplicaciones. Los modelos generativos adversarios (GANs) se han convertido en un estándar para muchas tareas, desde generación de moléculas candidatas, transferencia de estilo, detección de anomalías en imágenes médicas, y superresolución.

Cuando pensamos en llevar a producción modelos de aprendizaje profundo, necesitamos que estos sean robustos y equitativos. Actualmente, la mayoría de los modelos consideran condiciones ideales y suponen que los datos de producción provienen de la misma distribución que los de entrenamiento. Sin embargo, esto no suele ser el caso en las aplicaciones de la vida real. Por ejemplo, en un entorno clínico, podemos encontrar diferentes dispositivos de captura de imágenes, diversas poblaciones de pacientes, o condiciones médicas diferentes o desconocidas. Por otro lado, debemos evaluar las posibles disparidades en la evaluación o diagnóstico, ya que pueden trasladarse y amplificarse en nuestras soluciones de aprendizaje automático. 

### Contenidos

- Breve introducción al aprendizaje profundo y conceptos básicos.
  - Qué es aprendizaje profundo? En qué contextos sirve y en donde no.
  - Ciclo de experimentación con aprendizaje profundo (Datos, Diseño, Entrenamiento, Validación, Evaluación).
  - Conceptos básicos: Operadores y Técnicas de Optimización.

- Breve introducción a modelos generativos y conceptos básicos.
  - Cuando es útil usar modelos generativos?   En qué contextos sirve y en donde no.
  - Taxonomia de modelos generativos profundos.
  - Breve Introducción a GANs. Generación libre y condicional.
  - Cómo evaluar la performance de modelos generativos? Cómo evaluar los artefactos generados?
  - Limitaciones 

- Por qué es importante que nuestras soluciones sean robustas, generalizables y equitativas?
  - Discusiones en ambos ejemplos de aprendizaje profundo y generativos en particular.
  - Detección de valores atípicos.

[Presentaciones introductorias](https://drive.google.com/drive/folders/1-c2hkD_3bIulJLdWo_J9XdZoyHX7tGfk?usp=sharing)

### Referencias

- Navarro, P., Cintas, C., Lucena, M., Fuertes, J.M., Segura, R., Rueda, A., Ogayar-Anguita, C., González-José, R. and Delrieux, C., 2023. Iberianvoxel: Automatic Completion of Iberian Ceramics for Cultural Heritage Studies. International Conference on Joint Conferences on Artificial Intelligence, IJCAI 2023.
- Navarro, P., Cintas, C., Lucena, M., Fuertes, J.M., Segura, R., Delrieux, C. and González-José, R., 2022. Reconstruction of Iberian ceramic potteries using generative adversarial networks. Scientific Reports, 12(1), p.10644.
- Kim, H., Tadesse, G.A., Cintas, C., Speakman, S. and Varshney, K., 2022, March. Out-of-distribution detection in dermatology using input perturbation and subset scanning. In 2022 IEEE 19th International Symposium on Biomedical Imaging (ISBI) (pp. 1-4). IEEE.
- Cintas, C., Speakman, S., Tadesse, G.A., Akinwande, V., McFowland III, E. and Weldemariam, K., 2022. Pattern detection in the activation space for identifying synthesized content. Pattern Recognition Letters, 153, pp.207-213.
- Cintas, C., Das, P., Quanz, B., Tadesse, G.A., Speakman, S. and Chen, P.Y., 2022. Towards creativity characterization of generative models via group-based subset scanning . International Conference on International Joint Conferences on Artificial Intelligence, IJCAI 2022.
- Cintas, C., Speakman, S., Akinwande, V., Ogallo, W., Weldemariam, K., Sridharan, S. and McFowland, E., 2021. Detecting adversarial attacks via subset scanning of autoencoder activations and reconstruction error. In Proceedings of the Twenty-Ninth International Conference on International Joint Conferences on Artificial Intelligence (pp. 876-882).
- Navarro, P., Orlando, J.I., Delrieux, C. and Iarussi, E., 2021, February. SketchZooms: Deep Multiview Descriptors for Matching Line Drawings. In Computer Graphics Forum (Vol. 40, No. 1, pp. 410-423).


### Setup

#### Opción 1

`$ python -m venv genvenv`

`$ source genvenv/bin/activate`

`$ pip install -r requirements.txt`

`$ ipython kernel install --user --name=genvenv`

`$ python -m notebook`

#### Opción 2

`$ conda create --name curso python==3.9`

`$ conda activate curso`

`$ pip install -r requirements.txt`

`$ jupyter lab --port 8890`


