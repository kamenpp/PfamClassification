# PfamClassification

# Goal
Build a classifier that assigns a protein to the corresponding Pfam (Bateman et al., 2004, i.e., protein family). The point is that we would like to be able to predict the protein family and potential function of an unknown protein domain. This can be very beneficial for applications such as protein engineering and drug discovery.

The point is that we would like to be able to predict this from sequence data, instead of structure, since sequence data is more abundant and easier to obtain. One major challenge is that experimentally determining the protein family is time-consuming and expensive.

# Data
The Pfam seed random split dataset: https://www.kaggle.com/googleai/pfam-seed-random-split
It contains sequences of protein domains and their corresponding protein families. In addition, it contains the multiple sequence alignment for each domain. In total there are 17,929 protein families in Pfam v.32.0.

# Approach
Screening the literature for possible approaches shows that over the years there has been a variety of approaches that have been shown to work well. Examples include: 

### Alignment-based

1) pHMMER (https://www.ebi.ac.uk/Tools/hmmer/search/phmmer) - uses a profile hidden Markov models (pHMMs, probabilistic models that model the probability for each position of the chain to be occupied by an amino acid) of known protein families. When a protein sequence is submitted, pHMMER scores it against the entire database of pHMMs. If the sequence matches the conserved regions of a particular pHMM with a high score, it is assigned to the corresponding family.

2) BLASTp (https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE=Proteins) - first creates a set of short peptide sequences called "words" from the query sequence. These words are then used to search a database of protein sequences for matching words using a hashing algorithm. This initial search is designed to quickly filter out sequences that are not likely to be similar to the query. The remaining sequences that pass the initial filtering step are then aligned with the query sequence to find regions of similarity. Finally, the algorithm reports the top hits, ranked by similarity score.

### Alignment-free

Alignment-based protein family modeling requires a correct alignment of multiple sequences but it is hard to compute the multiple sequence alignment correctly and is essentially based on heuristics (Seo et al., 2018, 10.1093/bioinformatics/bty275). 

In addition, state-of-the-art alignment-based techniques cannot predict function for one-third of microbial protein sequences (Bileschi et al., 2022, 10.1038/s41587-021-01179-w).

1) K-mers (Vinga and Almeida, 2003) - use k amino acids, as features and use all possible k-mers as a feature vector. However, this loses the order information in biological sequences. They also don't capture the the biochemical properties, i.e. that substitution of a ceratin amino acid with an amino acid that has similar biochemical properties might not change the function of the domain at all.

2) Deep learning approaches - predominantly 1D CNN and RNN-based architectures (probably a matter of time for Transfomers to outperform them). RNNs (LSTMs, GRUs, etc) are, of course, very famous in NLP applications, but are notoriously hard and slow to train (vanishing/exploding gradients, hard to parallelise efficiently as the input to a hidden layer requires the output of the previous one). On the other hand, can be efficiently parallelised on a GPU, are fast to train and architectures like ResNet provide the chance to train very deep networks (like the ResNet that won the image classification challenge ImageNet in 2015 with 152 layers (!) https://arxiv.org/abs/1512.03385). One of the first CNN examples was DeepFam (Seo et al., 10.1093/bioinformatics/bty275). It features two convolutional layers, max pooling and two dense layers, followed by a softmax. The current state-of-the-art performance was achieved by Bileschi et al. (2019, 10.1038/s41587-021-01179-w) with ProtCNN. It features a 1D CNN with 5 consequent ResNet blocks, dense layer and softmax. The ResNet layers also include dilated convolutions to capture more distant relationships in sequence without sacrificing computational efficiency.

Thus, the approach that I have followed in my implementation follows that in ProtCNN, but because of limited computational resources, I cannot train such a deep network, with so many parameters, and hence I am sacrificing performance.

# Method



