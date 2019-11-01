# Accenture Challenge - Semana i 2019
Proyecto de Inteligencia Artificial y Visión computacional para resolver la situación del estacionamiento de alguna empresa o corporativo.

## Índice
- [Accenture Challenge - Semana i 2019](#accenture-challenge---semana-i-2019)
  - [Índice](#%c3%8dndice)
    - [Solución](#soluci%c3%b3n)
    - [Contenido](#contenido)
      - [Vehículos](#veh%c3%adculos)
      - [Placas](#placas)
      - [Caras](#caras)
      - [Estacionamiento](#estacionamiento)
    - [Equipo](#equipo)


### Solución
Nuestra solución consiste en la detección de la entrada de carros, detectando el color de este, sus placas, así como la toma de la imagen de la identificación del conductor.

### Contenido
* [my-app](https://github.com/DanielSGA/accenture-challenge/tree/master/my-app): contiene lo relacionado con la web app desarrollada en react para mostrar una aplicación de búsqueda de placas con información del carro y su dueño.
* [face-detection](https://github.com/DanielSGA/accenture-challenge/tree/master/face-detection): contiene el código para la detección de cara de una identificación que a la vez almacena.


#### Vehículos
Cuando un vehículo va ingresando al estacionamiento, una cámara que graba, detecta que lo que se aproxima es un carro, detecta su color y toma una foto de la placa.

#### Placas
Al tener una fotografía de la placa, se detecta el texto y se guardan las imágenes de cada carro con el nombre de su placa correspondiente.

#### Caras
Utilizando OPENCV en Python, se toman las fotografías para luego tomar el lado izquierdo de esta imagen y detectarla mediante Haarcascade.

![face](face-detection/detected/face_412.jpg)
`$ python3 detect-crop-face.py`
#### Estacionamiento

### Equipo
* Jorge (IMT)
* Sheccid Itzel Sánchez (I2D)
* Daniel Alejandro Saldaña (ITC)
* Flor Esthela Barbosa A01281460 (ITC)