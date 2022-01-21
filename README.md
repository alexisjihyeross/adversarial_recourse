# Learning Models for Actionable Recourse

This repository contains code for our paper, [Learning Models for Actionable Recourse](https://proceedings.neurips.cc/paper/2021/file/9b82909c30456ac902e14526e63081d4-Paper.pdf).

## Citation
```bibtex
@inproceedings{
  ross2021learning,
  title={Learning Models for Actionable Recourse},
  author={Alexis Ross and Himabindu Lakkaraju and Osbert Bastani},
  booktitle={Advances in Neural Information Processing Systems},
  editor={A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
  year={2021},
  url={https://openreview.net/forum?id=JZK9uP4Fev}
}
```
## Installation

1.  Clone the repository.
    ```bash
    git clone https://github.com/alexisjihyeross/adversarial_recourse
    cd adversarial_recourse
    ```

2.  [Download and install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

3.  Create a Conda environment.

    ```bash
    conda create -n adversarial_recourse python=3.7
    ```
 
4.  Activate the environment.

    ```bash
    conda activate adversarial_recourse
    ```
    
5.  Download the requirements.

    ```bash
    pip3 install -r requirements.txt
    ```
    
  To run the linear approximation experiments, you will also need to download the appropriate optimization software for using the `recourse` library. 
  See [here](https://github.com/ustunb/actionable-recourse#requirements) for more details.
