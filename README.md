# Implementation of `High-Quality Motion Deblurring from a Single Image (SIGGRAPH 2008)`
[Original Paper](http://www.cse.cuhk.edu.hk/%7Eleojia/projects/motion_deblurring/index.html)

Para rodar o programa execute o arquivo main.py
Parametros que podem ser ajustados 

I: alterar o caminho da imagem original a ser convolucionada
f: alterar o kernel inicial (tamanho e angulacao)
n_rows: tamanho que ira ser selecionado para computar a matriz A. Nos nossos experimentos os melhores resultados foram obtidos com n_rows = min(width, height)/2 - 12
