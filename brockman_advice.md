Quora link: https://www.quora.com/What-are-the-best-ways-to-pick-up-Deep...


If you want to read one main resource... the Goodfellow, Bengio, Courville book (available for free from http://www.deeplearningbook.org/) is an extremely comprehensive survey of the field. It contains essentially all the concepts and intuition needed for deep learning engineering (except reinforcement learning).

If you'd like to take courses... Pieter Abbeel and Wojciech Zaremba suggest the following course sequence:

- Linear Algebra — Stephen Boyd’s EE263 (Stanford) - Neural Networks for Machine Learning — Geoff Hinton (Coursera) - Neural Nets — Andrej Karpathy’s CS231N (Stanford) - Advanced Robotics (the MDP / optimal control lectures) — Pieter Abbeel’s CS287 (Berkeley) - Deep RL — John Schulman’s CS294-112 (Berkeley)

(Pieter also recommends the Cover & Thomas information theory and Nocedal & Wright nonlinear optimization books).

If you'd like to get your hands dirty... Ilya Sutskever recommends implementing simple MNIST classifiers, small convnets, reimplementing char-rnn, and then playing with a big convnet. Personally, I started out by picking Kaggle competitions (especially the "Knowledge" ones) and using those as a source of problems. Implementing agents for OpenAI Gym (or algorithms for the set of research problems we’ll be releasing soon) could also be a good starting place.

