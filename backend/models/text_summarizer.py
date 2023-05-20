# python3 text_summarizer.py

from summarizer import Summarizer,TransformerSummarizer

body = '''
Certainty Factor (CF) is an important component of an Expert System (ES) as it allows the ES to reason under uncertainty and express its level of confidence in the conclusions it draws. The CF approach allows the ES to reason based on incomplete or uncertain information, which is common in many real-world problems, just like the one in the above journal article.

The CF approach assigns a numerical value to each rule in the ES, representing the degree of certainty or confidence that the rule is true. The CF value can range from -1 (completely false) to +1 (completely true), with 0 indicating a neutral or uncertain position. The CF values of multiple rules can be combined to arrive at an overall conclusion with a higher level of certainty and can be updated as new evidence is gathered, allowing the ES to adapt and refine its conclusions over time. The application of CF to an expert system requires a few rules in the form of variables and weights predefined by the domain expert, with notation certainty factors as below:

CF[h,e] = MB[h,e] - MD[h,e]

and confidence propagation for a rule with one premise is obtained by the below:

CF(Hypothesis, Evidence) = CF(Evidence) * CF(Rule)

One advantage of the CF approach is its ability to handle uncertain or incomplete information. By assigning numerical values to the degree of belief or confidence as mentioned, the ES can reason and make decisions even with incomplete or uncertain data, boosting the ES’ uncertainty handling. The CF approach can also be updated as new evidence is gathered, allowing the ES to adapt and refine its conclusions over time.

Another method that can be used as the CF for the proposed ES is the Bayesian Probability. Bayesian Probability is a mathematical approach that allows for the calculation of the probability of an event based on prior knowledge or experience. In the context of the above mentioned ES, Bayesian Probability can be used to calculate the probability of a heart disease diagnosis based on the patient's symptoms and medical history.

Before any evidence is considered, the ES assigns initial probabilities to different hypotheses or conclusions. These probabilities represent the belief in each hypothesis based on prior knowledge or experience. For example, in the case of a heart disease diagnosis ES, prior probabilities could represent the likelihood of different heart conditions based on their prevalence in the population. As new evidence is gathered, the ES calculates likelihood ratios that quantify the strength of the evidence supporting or contradicting each hypothesis. The prior probabilities are updated using the likelihood ratios to calculate the posterior probabilities, which represent the updated belief in each hypothesis after considering the evidence. This update is performed using Bayes' theorem. Based on the posterior probabilities, the ES can make decisions or draw conclusions. The hypothesis with the highest posterior probability is often selected as the most likely or preferred conclusion.

Bayesian Probability is particularly useful in situations where there is limited data available or where the data is noisy or uncertain. This works well for this type of ES as patient’s medical data are rather noisy compared to digital data and is limited in a sense that such data is hard to be obtained. By incorporating prior knowledge or experience, the ES can make more accurate predictions even with incomplete or uncertain data. As more data becomes available, the ES can update its beliefs accordingly.

Bayesian Probability is also a good choice for this ES as it is quite robust when it comes to dealing with uncertainty. If the data is uncertain or ambiguous, the ES can capture this uncertainty through prior probabilities and likelihood ratios. The ES can then update its beliefs based on the available evidence, resulting in updated posterior probabilities that reflect the updated level of uncertainty. In some cases where available evidence may not lead to a definitive or conclusive result, Bayesian Probability can handle uncertain results by providing a probabilistic representation of the conclusions. Instead of producing a single, deterministic answer, the ES can generate a range of probabilities for different hypotheses that are equally plausible.

In sum, CF is a good choice for the above mentioned ES. An alternative for CF would be Bayesian Probability, which is much more robust and better at dealing with limited and noisy data as well as uncertainty.

'''

# Bert (~1.8gb)
# bert_model = Summarizer()
# bert_summary = ''.join(bert_model(body, min_length=60))
# print(bert_summary)

# GPT-2: Better (~1.9gb)
# GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
# full = ''.join(GPT2_model(body, min_length=60))
# print(full)

# XLNet: Best (~1gb)
model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
full = ''.join(model(body, min_length=60))
print(full)