# Uncertainty Estimation for GNNs based on SPDEs

This repository provides the source code for **uncertainty estimation in Graph Neural Networks (GNNs)** on both **static** and **dynamic graphs based on specific SPDEs**. 

## Datasets

### Static Graph Datasets
 **Cora:** Downloadable from https://github.com/kimiyoung/planetoid/tree/master/data
 
 **CiteSeer:** Downloadable from https://github.com/kimiyoung/planetoid/tree/master/data

   **Pubmed:**  Downloadable from https://github.com/kimiyoung/planetoid/tree/master/data

   **OGBN-Arxiv:** Downloadable from https://ogb.stanford.edu/docs/nodeprop/

  **Amazon-Computers:** Downloadable from https://github.com/shchur/gnn-benchmark/
  
### Dynamic Graph Datasets

 **BC-OTC:** Downloadable from http://snap.stanford.edu/data/soc-sign-bitcoin-otc.html
 
 **Reddit:** Downloadable from http://snap.stanford.edu/data/soc-RedditHyperlinks.html

   **UCI:** Downloadable from http://konect.uni-koblenz.de/networks/opsahl-ucsocial 

   **AS:** Downloadable from http://snap.stanford.edu/data/as-733.html

  **Elliptic:** Downloadable from https://www.kaggle.com/ellipticco/elliptic-data-set 

  **Brain:** Downloadable from https://www.dropbox.com/sh/33p0gk4etgdjfvz/AACe2INXtp3N0u9xRdszq4vua?dl=0
  
## Code Usage

For static graphs: python -u main.py  --use_bn --dataset --ood_type  --lr  --weight_decay --input_dropout --device 

For dynamic graphs: python main_dgnn.py --config_file --OOD 

## References

[1] Xixun Lin, Wenxiao Zhang, Fengzhao Shi, Chuan Zhou, Lixin Zou, Xiangyu Zhao, Dawei Yin, Shirui Pan, Yanan Cao. Graph Neural Stochastic Diffusion for Estimating Uncertainty in Node Classification. International Conference on Machine Learning (ICML 2024)

[2] Xixun Lin, Zhiheng Zhou, Yanan Cao, Chuan Zhou, Nan Sun, Ge Zhang, Xiangyu Zhao, Peng Zhang, Peilin Zhao, Shirui Pan, Philip S. Yu. Graph Stochastic Jump-Diffusion Equation for Uncertainty Estimation on Dynamic Graphs. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI, under review)
