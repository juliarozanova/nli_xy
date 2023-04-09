# NLI-XY
Dataset and experimental code for the paper ["Decomposing Natural Logic Inferences in Neural NLI"](https://arxiv.org/abs/2112.08289).


```
@inproceedings{rozanova-etal-2022-decomposing,
    title = "Decomposing Natural Logic Inferences for Neural {NLI}",
    author = "Rozanova, Julia  and
      Ferreira, Deborah  and
      Thayaparan, Mokanarangan  and
      Valentino, Marco  and
      Freitas, Andre",
    booktitle = "Proceedings of the Fifth BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.blackboxnlp-1.33",
    pages = "394--403",
    abstract = "In the interest of interpreting neural NLI models and their reasoning strategies, we carry out a systematic probing study which investigates whether these modelscapture the crucial semantic features central to natural logic: monotonicity and concept inclusion.Correctly identifying valid inferences in downward-monotone contexts is a known stumbling block for NLI performance,subsuming linguistic phenomena such as negation scope and generalized quantifiers.To understand this difficulty, we emphasize monotonicity as a property of a context and examine the extent to which models capture relevant monotonicity information in the vector representations which are intermediate to their decision making process.Drawing on the recent advancement of the probing paradigm,we compare the presence of monotonicity features across various models.We find that monotonicity information is notably weak in the representations of popularNLI models which achieve high scores on benchmarks, and observe that previous improvements to these models based on fine-tuning strategies have introduced stronger monotonicity features together with their improved performance on challenge sets.",
}
```
