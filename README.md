# keras-emoji-embeddings
A Jupyter notebook that implements an approach to generating embeddings for Unicode emoji from their descriptions, based on the work by Ben Eisner and his colleagues at the University College London Machine Reading Group [[1]](https://arxiv.org/abs/1609.08359).

## Requirements

* Python 3.5.2
* jupyter 4.2.0

## Package dependencies

* numpy 1.11.1
* pandas 0.18.1
* Keras 1.1.2
* scikit-learn 0.17.1
* h5py 2.6.0
* hdf5 1.8.17

## Implementation notes

Although the approach taken to generating embeddings is based on that taken in the paper, it is different in a number of respects.

* We use GloVe [[2]](http://nlp.stanford.edu/pubs/glove.pdf) (specifically, the 300-dimensional version of the 6-billion tokens derived by crawling Wikipedia 2014 and Gigaword 5, provided as glove.6b.300d.txt) as the source of fixed word embeddings, as opposed to the 300-dimensional Google News word2vec embeddings used in [[1]](https://arxiv.org/abs/1609.08359).
* Following a suggestion by Eisner [[3]](#eisner-personal-communication), we use a bidirectional GRU to compute a 300-dimensional embedding for the emoji description, as opposed to the paper's straightforward summation of the embeddings of the constituent terms.
* Instead of taking the sigmoid of the dot product of the emoji and description embeddings, we instead concatenate them into a 600-dimensional vector and then pass that up through several densely-connected layers to a 2-dimensional softmax layer.

The model architecture is shown below:

![[Keras architecture for Unicode emoji embedding generation]](emoji_emb_arch.png)

This architecture takes a good deal of inspiration from the emerging architectural patterns in deep learning for natural language inference, e.g. as described in [[4]](https://arxiv.org/abs/1607.04853v2) and [[5]](https://explosion.ai/blog/deep-learning-formula-nlp). The motivation for the differences is that we hope to exploit these architectural patterns to create embeddings for terms from controlled vocabularies, where descriptions will be longer than those associated with Unicode emoji.

## Results

Interestingly, in spite of the differences described above, a t-SNE visualization of the computed embeddings shows clusterings recognizable from those in the visualization published in [[1]](https://arxiv.org/abs/1609.08359):

![[A t-SNE visualization of the computed embeddings]](emoji_emb_viz.png)

## Usage

Simply run the notebook using the standard Jupyter command:

    $ jupyter notebook

Apart from running the notebook, one can view a t-SNE visualization of the computed embeddings by running the following command:

    $ python visualize.py

## License

MIT. See the LICENSE file for the copyright notice.

## References

[[1]](https://arxiv.org/abs/1609.08359) Ben Eisner, Tim Rocktäschel, Isabelle Augenstein, Matko Bošnjak, and Sebastian Riedel. “emoji2vec: Learning Emoji Representations from their Description,” in Proceedings of the 4th International Workshop on Natural Language Processing for Social Media at EMNLP 2016 (SocialNLP at EMNLP 2016), November 2016.

[[2]](http://nlp.stanford.edu/pubs/glove.pdf) Jeffrey Pennington, Richard Socher, and Christopher D. Manning. "GloVe: Global Vectors for Word Representation," in Proceedings of the 2014 Conference on Empirical Methods In Natural Language Processing (EMNLP 2014), October 2014.

[[3]](#eisner-personal-communication) Ben Eisner. Private communication, 22 November 2016.

[[4]](https://arxiv.org/abs/1607.04853v2) Anirban Laha and Vikas Raykar. "An Empirical Evaluation of various Deep Learning Architectures for Bi-Sequence Classification Tasks," in Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers, p. 2762–2773, Osaka, Japan, 11-17 December 2016.

[[5]](https://explosion.ai/blog/deep-learning-formula-nlp) Matthew Honnibal. "Embed, encode, attend, predict: The new deep learning formula for state-of-the-art NLP models", 10 November 2016. Retrieved at https://explosion.ai/blog/deep-learning-formula-nlp on 1 December 2016.
