# Fair Graph Datasets

This repository provides `torch_geometric` dataset classes for graph datasets used in Graph-Fairness literature [1].

| Dataset    | Nodes | Edges  | Sensitive Attribute |
|------------|-------|--------|---------------------|
| NBA        | 403   | 10,621 | nationality         |
| UNC28      | 3,111 | 73,230 | gender              |
| Oklahoma97 | 4,018 | 65,287 | gender              |

## NBA Dataset
A graph of NBA players during the 2016-2017 season [2].

```python
dataset = NBADataset(data_path)
sensitive_attribute = dataset.get_sensitive_attribute()
```

## Collegiate Dataset
Dataset of friend connections of college students in a social network [3].

### UNC28
A graph of connections in a social network of students from the University of
North Carolina.

```python
dataset = CollegiateSocNet(data_path, "unc28")
sensitive_attribute = dataset.get_sensitive_attribute()
```

### Oklahoma97
A graph of connections in a social network of students from the University
of Oklahoma.

```python
dataset = CollegiateSocNet(data_path, "oklahoma97")
sensitive_attribute = dataset.get_sensitive_attribute()
```


## References

[1] Zichong Wang, Charles Wallace, Albert Bifet, Xin Yao, and Wenbin Zhang. fg2an:
Fairness-aware graph generative adversarial networks. In Danai Koutra, Claudia Plant,
Manuel Gomez Rodriguez, Elena Baralis, and Francesco Bonchi, editors, Machine Learn-
ing and Knowledge Discovery in Databases: Research Track, pages 259–275, Cham, 2023.
Springer Nature Switzerland.

[2] Dai, E., Wang, S.: Say no to the discrimination: Learning fair graph neural networks
with limited sensitive attribute information. In: Proceedings of the 14th ACM
International Conference on Web Search and Data Mining. pp. 680–688 (2021)

[3] Red, V., Kelsic, E.D., Mucha, P.J., Porter, M.A.: Comparing community structure
to characteristics in online collegiate social networks. SIAM Rev. 53(3), 526–543
(2011)
