# AI_HW_5

Mesterséges Intelligencia házi feladat #5

_Perceptrons_

The __readme__ is also available in English [here](#tasks).

## Feladatok
AND logikai függvény/kapu megtanítása perceptronnak. 
  * [x] Tantítás
  * [x] Döntési felület kirajzolása  
  
H, I betűk megtanítása perceptronnak 3x3-as képek esetén
  * [x] H, I betűk megtanítása
  * [x] opcionális: T, O betűk megtanítása
  
A háló szépen megtanulja a betűket, illetve pár módosított betűt is felismer, mint például a T-t, ha hiányzik a felső vonal középső pixele, Az O-t, ha hiányzik a középső sáv pixelei stb. A pontossága állítható az `error_rate` változóval.

A programok Python nyelven lettek írva, 3.7-es verzióval.

![Example1](https://github.com/naghim/AI_HW_5/blob/master/example1.PNG)

## Tasks
Teach AND logic gate to a perceptron.
  * [x] Teaching
  * [x] Drawing the decision surface
  
Teach H, I letters to a perceptron (3x3 images)
  * [x] Teaching H, I
  * [x] Optional: Teach T and O 

The network learns the letters well and recognizes a few modified letters, such as T if the pixel in the horizontal line is missing, O if the pixels are missing in the 2nd row, and so on. The accuracy can be adjusted with the `error_rate` variable.

The programs were written in Python, version 3.7.
