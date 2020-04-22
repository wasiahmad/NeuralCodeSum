## A Transformer-based Approach for Source Code Summarization
Official implementation of our ACL 2020 paper

### Installing C2NL

You may consider installing the C2NL package. C2NL requires Linux and Python 3.6 or higher. It also requires installing PyTorch version 1.3. Its other dependencies are listed in requirements.txt. CUDA is strongly recommended for speed, but not necessary.

Run the following commands to clone the repository and install C2NL:

```
git clone https://github.com/wasiahmad/NeuralCodeSum.git
cd NeuralCodeSum; pip install -r requirements.txt; python setup.py develop
```

### Training/Testing Models

We provide a RNN-based sequence-to-sequence (Seq2Seq) model implementation along with our Transformer model. To perform training and evaluation, first go the scripts directory associated with the target dataset.

```
$ cd  scripts/DATASET_NAME
```

Where, choices for DATASET_NAME are ["java", "python"].

To train/evaluate a model, run:

```
$ bash script_name.sh GPU_ID MODEL_NAME
```

For example, to train/evaluate the transformer model, run:

```
$ bash transformer.sh 0,1 code2jdoc
```

#### Generated log files

While training and evaluating the models, a list of files are generated inside a `tmp` directory. The files are as follows.

- **MODEL_NAME.mdl**
  - Model file containing the parameters of the best model.
- **MODEL_NAME.mdl.checkpoint**
  - A model checkpoint, in case if we need to restart the training.
- **MODEL_NAME.txt**
  - Log file for training.
- **MODEL_NAME.json**
  - The predictions and gold references are dumped during validation.
- **MODEL_NAME_test.txt**
  - Log file for evaluation (greedy).
- **MODEL_NAME_test.json** 
  - The predictions and gold references are dumped during evaluation (greedy).
- **MODEL_NAME_beam.txt**
  - Log file for evaluation (beam).
- **MODEL_NAME_beam.json**
  - The predictions and gold references are dumped during evaluation (beam).

**[Structure of the JSON files]** Each line in a JSON file is a JSON object. An example is provided below.

```json 
{
    "id": 0,
    "code": "private int current Depth ( ) { try { Integer one Based = ( ( Integer ) DEPTH FIELD . get ( this ) ) ; return one Based - NUM ; } catch ( Illegal Access Exception e ) { throw new Assertion Error ( e ) ; } }",
    "predictions": [
        "returns a 0 - based depth within the object graph of the current object being serialized ."
    ],
    "references": [
        "returns a 0 - based depth within the object graph of the current object being serialized ."
    ],
    "bleu": 1,
    "rouge_l": 1
}
```

#### Running experiments on CPU/GPU/Multi-GPU

- If GPU_ID is set to -1, CPU will be used.
- If GPU_ID is set to one specific number, only one GPU will be used.
- If GPU_ID is set to multiple numbers (e.g., 0,1,2), then parallel computing will be used.

### Acknowledgement

We borrowed and modified code from [DrQA](https://github.com/facebookresearch/DrQA), [OpenNMT](https://github.com/OpenNMT/OpenNMT-py). We would like to expresse our gratitdue for the authors of these repositeries.


### Citation

```
@inproceedings{zhang2018graph,
 author = {Ahmad, Wasi Uddin and Chakraborty, Saikat and Ray, Baishakhi and Chang, Kai-Wei},
 booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)},
 title = {A Transformer-based Approach for Source Code Summarization},
 year = {2020}
}
```

