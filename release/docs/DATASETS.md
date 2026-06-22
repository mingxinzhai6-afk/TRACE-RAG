# Dataset Links

This release targets two benchmark datasets used in the paper experiments.
The local layout expected by the backend is:

```text
datasets/
  Popqa/
    Corpus.json
    Question.json
  musique/
    Corpus.json
    Question.json
```

## Official Dataset Links

| Dataset | Link | Notes |
| --- | --- | --- |
| PopQA | https://hf.co/datasets/akariasai/PopQA | Hugging Face dataset card for the PopQA benchmark. |
| MuSiQue | https://hf.co/datasets/dgslibisey/MuSiQue | Hugging Face dataset card for the MuSiQue benchmark. |

## Practical Notes

- The release assumes the datasets are already placed under the DIGIMON root
  in the layout shown above.
- The experiments in this repository use a fixed 200-sample subset for paper
  reporting.
- The 200-sample subset is about 14.3% of PopQA's 1399 examples and about
  6.7% of MuSiQue's 3000 examples.
