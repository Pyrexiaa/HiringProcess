# python3 text_summarizer.py
from summarizer import TransformerSummarizer
def summarize_text(input: list) -> list:
    # XLNet: Best (~1gb)
    re = None
    if len(input) > 0:
        re = []
        model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
        for i in input:
            re.append(''.join(model(i, min_length=60)).strip())
    return re



if __name__ == '__main__':
    test = ['''
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
        ,
        '''
        Taylor's University (commonly referred to as Taylor's) is a private university in Subang Jaya, Selangor, Malaysia. It is often regarded as Malaysia's top private university in Malaysia based on the QS World University Rankings.[1][2]

        It was founded in 1969 as a college, was awarded university college status in 2006, and university status in 2010.

        Taylor's University is a member of the Taylor's Education Group, which also includes British University Vietnam, Taylor’s College, Garden International School, Nexus International School, Australian International School Malaysia, and Taylor’s International School.

        History[edit]
        In 2006, Taylor’s was granted ‘University College’ status,[3] which resulted in two distinct identities under the tertiary arm of the brand – Taylor’s College and Taylor’s University College.

        Work commenced to build the RM450 million Taylor's University Lakeside Campus, in Subang Jaya, in early 2007 and was completed in January 2010.

        Officially awarded as a full-fledged university status in September 2010[4] by the Ministry of Higher Education Malaysia, Taylor’s is now one of the nation’s leading private higher education institutions.[5]

        Lakeside Campus[edit]
        The Taylor's Lakeside Campus was completed in January 2010 and is set on 27 acres of tropical greenery, surrounded by a 5.5-acre man-made lake.
        Completed in January 2010 and set on 27-acres of tropical greenery, the integrated purpose-built campus surrounds a 5.5-acre man-made lake, as well as a landscape of water plants, trees, and flowering shrubs.
        Taylor's University established its Lakeside Campus in 2010 to accommodate the ever-expanding number of students. On the grounds of the campus itself lies Syopz, a commercial block with restaurants and shops. U Residence, the official accommodation for students, is also located there.[6]

        In 2011, Taylor’s received the Special Honour Award from the Institute of Landscape Architects Malaysia in Category 3: Professional Awards in Landscape Design and Planning.[7] The Lakeside Campus was awarded in all three building categories of Interior Design, Architecture and Landscape.
        '''
        ,
        '''
        A transformer is a deep learning model. It is distinguished by its adoption of self-attention, differentially weighting the significance of each part of the input (which includes the recursive output) data. It is used primarily in the fields of natural language processing (NLP)[1] and computer vision (CV).[2]

        Like recurrent neural networks (RNNs), transformers are designed to process sequential input data, such as natural language, with applications towards tasks such as translation and text summarization. However, unlike RNNs, transformers process the entire input all at once. The attention mechanism provides context for any position in the input sequence. For example, if the input data is a natural language sentence, the transformer does not have to process one word at a time. This allows for more parallelization than RNNs and therefore reduces training times.[1]

        Transformers were introduced in 2017 by a team at Google Brain[1] and are increasingly becoming the model of choice for NLP problems,[3] replacing RNN models such as long short-term memory (LSTM).[4] Compared to RNN models, transformers are more amenable to parallelization, allowing training on larger datasets. This led to the development of pretrained systems such as BERT (Bidirectional Encoder Representations from Transformers) and the original GPT (generative pre-trained transformer), which were trained with large language datasets, such as the Wikipedia Corpus and Common Crawl, and can be fine-tuned for specific tasks.[5][6]

        Background[edit]
        Before transformers, most state-of-the-art NLP systems relied on gated RNNs, such as LSTMs and gated recurrent units (GRUs), with added attention mechanisms. Transformers also make use of attention mechanisms but, unlike RNNs, do not have a recurrent structure. This means that provided with enough training data, attention mechanisms alone can match the performance of RNNs with attention.[1]

        The terms "query", "key", "value" are borrowed from key–value databases.

        Previous work[edit]
        In 1992, Jürgen Schmidhuber published the fast weight controller as an alternative to RNNs that can learn "internal spotlights of attention,"[7] and experimented with using it to learn variable binding.[8]

        In a fast weight controller, a feedforward neural network ("slow") learns by gradient descent to control the weights of another neural network ("fast") through outer products of self-generated activation patterns called "FROM" and "TO" which corresponds to "key" and "value" in the attention mechanism.[9] This fast weight is applied to queries. The attention mechanism may be obtained by interposing a softmax operator and three linear operators (one for each of query, key, and value).[9][10]
        '''
        ,
        '''
        React (also known as React.js or ReactJS) is a free and open-source front-end JavaScript library[3][4] for building user interfaces based on components. It is maintained by Meta (formerly Facebook) and a community of individual developers and companies.[5][6][7]

        React can be used to develop single-page, mobile, or server-rendered applications with frameworks like Next.js. Because React is only concerned with the user interface and rendering components to the DOM, React applications often rely on libraries for routing and other client-side functionality.[8][9]
        Notable features[edit]
        Declarative[edit]
        React adheres to the declarative programming paradigm. Developers design views for each state of an application, and React updates and renders components when data changes. This is in contrast with imperative programming.[10]

        Components[edit]
        React code is made of entities called components. These components are modular and reusable. React applications typically consist of many layers of components. The components are be rendered to a root element in the DOM using the React DOM library. When rendering a component, values are passed between components through props (short for "properties"). Values internal to a component are called its state.[11]

        The two primary ways of declaring components in React are through function components and class components.
        React Hooks[edit]
        On February 16, 2019, React 16.8 was released to the public, introducing React Hooks.[12] Hooks are functions that let developers "hook into" React state and lifecycle features from function components.[13] Notably, Hooks do not work inside classes — they let developers use more features of React without classes.[14]

        React provides several built-in Hooks such as useState,[15] useContext, useReducer , useMemo and useEffect.[16] Others are documented in the Hooks API Reference.[17] useState and useEffect, which are the most commonly used, are for controlling state and side effects respectively.

        Rules of hooks[edit]
        There are two rules of Hooks[18] which describe the characteristic code patterns that Hooks rely on:

        "Only Call Hooks at the Top Level" — Don't call hooks from inside loops, conditions, or nested statements so that the hooks are called in the same order each render.
        "Only Call Hooks from React Functions" — Don't call hooks from plain JavaScript functions so that stateful logic stays with the component.
        Although these rules can't be enforced at runtime, code analysis tools such as linters can be configured to detect many mistakes during development. The rules apply to both usage of Hooks and the implementation of custom Hooks,[19] which may call other Hooks.'''
        ]
        
    
    result = summarize_text("My name is Lim Jia Yu, currently a sophomore in Universiti Malaya, Malaysia pursuing Bachelor of Computer Science, majoring in Artificial Intelligence. I am a tenacious and industrious individual that embraces hindrances, as the key to my life is accepting challenges. I am always prepared to admit my own imperfection, and willing to learn from anyone for continuous self-improvement. I would also acknowledge myself as a sporty person as I love playing sports in my free time especially basketball and badminton, it has been undetachable from my life. The main reason that prompted me to choose this degree path is I am fascinated to envisage how Artificial Intelligence (AI) is going to revolutionize our future. No one could ever stop learning in this constantly-developing world, in order to prevail over the obstacles in life and to bring a better future to the incoming generation. Hence, I am always inquisitive in discovering a better computer science-related skill set in myself, namely AI. Speaking of my career specific goals, as I have not completely specialized in Artificial Intelligence fields such as Image Processing, Computer Vision etc, I am still unsure of my further career or study pathway. However, I would definitely dive deep into AI, utilize what I have learnt in creating impacts to the world that I lived in. I would not expect a great start, but continuous improvement in my career is a definite answer to me. Most importantly, work-life balance is what I will strive for, landing in a management position eventually. Through this exchange, I hope to inspire myself in gaining different perspectives regarding my life as well as my future career by experiencing an absolutely different lifestyle in a distinct environment. Staying inside a bubble for too long might enclose my creativity and possibility in achieving something that I might never ever imagined before, hence I believe this exchange will act as a powerful stop for me to unlearn and relearn knowledge that I obtained or going to obtain. Besides, cultural exchange is what I am always interested in, as it paints the world, fabricate a colorful life for all of us to live in. Understanding different cultures forms a more comprehensive personality in me, allowing me to work and adapt in different places easily. Last but not least, undeniably, my financial situation is a big obstacle to me in applying for this exchange. Although my family in Malaysia is still counted as a middle-class family, the fact that the currency rate between MYR and CHF is nearly fivefold has definitely increased my family’s financial pressure. Considering that my parents are the only source of income for the 3 children in this family including me, while my little brother is studying in a private university which costs a lot more than me who studied in a public university, every huge financial decision that I made is significant in determining the living quality of my family. After a solemn discussion with my parents who always gave me full support and invested all of their hope in my educational pathway, they agreed with my dream to study abroad as an exchange student. My parents always have been giving the best to all of us, no matter how big the storm is while they are working outside, sunshine is the only thing that they will present to us. As a teenager that can think rationally, I clearly understand the consequences and responsibilities that they will have to bear in order to support my decision. Therefore, if I was given this scholarship, it would definitely reduce a lot on my parents’ shoulders. This scholarship will be allocated properly, only for my study and living usage in Switzerland.")
    for i in result:
        print(i, "\n\n")
