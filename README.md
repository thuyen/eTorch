## Simple CNN for pupil detection
Download [the data](https://www.cl.cam.ac.uk/research/rainbow/projects/pupiltracking/) and put it in the input directory.
Sample results will come soon.
````
input
  p1-left/  
  p1-right/  
  p2-left/  
  p2-right/
src
  main.py
  data.py
  model.py
````

````
python main.py --data ../input/ --train_list train_data.csv
python main.py --data ../input/ --valid_list valid_data.csv --evaluate \
  --resume ckpts/model_300.pth.tar --out_file preds.npy
`````
