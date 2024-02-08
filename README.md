# ODINN

## Overview
Here we provide the implementation of ODINN in TensorFlow. The repository is organised as follows:
- `data/` contains datasets Cora, Pubmed, Amazon Computers, Amazon Photo, MS Academic CS (Coauthor CS), MS Academic Physics (Coauthor Physics);
- `models/` contains the implementation of the ODNN;
- `utils/` contains:
    * an implementation of the aggregation by the DeGroot model and the Friedkin-Johnsen model (`layers.py`);
    * preprocessing subroutines (`process.py`);

Finally, `bash run_train` execute the experiments.


## Dependencies

The script has been tested running under Python 3.7.9, with TensorFlow version as:
- `tensorflow-gpu==1.13.1`

In addition, CUDA 11.1 has been used.


## License
MIT
