# MMgraph

## Dependencies
+ python 3.7  
+ pytorch 1.4.0    
+ pytorch_pretrained
+ numpy 1.16.2

## Datasets
官方提供数据集 
+ train.json
+ dev.json
+ company_2_code_sub.txt

### Decompress and composite the data
```bash
$ python ./get_input.py
```
### Train

关于学习率等在脚本中设置

```bash
$ python ./train.py
```
### Pseudo Semi Supervised

关于学习率等在脚本中设置

```bash
$ python ./pseudo.py
```

## Results and trained model

```bash
$ python ./get_result.py
```