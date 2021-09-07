# MLFont: Few-Shot Chinese Font Generation via Deep Meta-Learning
PyTorch implementation of MLFont | [paper](https://dl.acm.org/doi/10.1145/3460426.3463606)

# Abstract
The automatic generation of Chinese fonts is challenging due to the large quantity and complex structure of Chinese characters. When there are insufficient reference samples for the target font, existing deep learning-based methods cannot avoid overfitting caused by too few samples, resulting in blurred glyphs and incomplete strokes. To address these problems, this paper proposes a novel deep meta-learning-based font generation method (MLFont) for few-shot Chinese font generation, which leverages existing fonts to improve the generalization capability of the model for new fonts. Existing deep meta-learning methods mainly focus on few-shot image classification. To apply meta-learning to font generation, we present a meta-training strategy based on Model Agnostic Meta-Learning (MAML) and a task organization method for font generation. The meta-training makes the font generator easy to fine-tune for new font generation tasks. Through random font generation tasks and extraction of glyph content and style separately, the font generator learns the prior knowledge of character structure in the meta-training stage, and then quickly adapts to the generation of new fonts with a few samples by fine-tuning of adversarial training. Extensive experiments demonstrate that our method outperforms the state-of-the-art methods with more complete strokes and less noise in the generated character images.

# Prerequisites
python > 3.6
pytorch == 1.2
