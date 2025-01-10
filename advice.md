# Brockman

Quora link: https://www.quora.com/What-are-the-best-ways-to-pick-up-Deep...


If you want to read one main resource... the Goodfellow, Bengio, Courville book (available for free from http://www.deeplearningbook.org/) is an extremely comprehensive survey of the field. It contains essentially all the concepts and intuition needed for deep learning engineering (except reinforcement learning).

If you'd like to take courses... Pieter Abbeel and Wojciech Zaremba suggest the following course sequence:

- Linear Algebra — Stephen Boyd’s EE263 (Stanford) - Neural Networks for Machine Learning — Geoff Hinton (Coursera) - Neural Nets — Andrej Karpathy’s CS231N (Stanford) - Advanced Robotics (the MDP / optimal control lectures) — Pieter Abbeel’s CS287 (Berkeley) - Deep RL — John Schulman’s CS294-112 (Berkeley)

(Pieter also recommends the Cover & Thomas information theory and Nocedal & Wright nonlinear optimization books).

If you'd like to get your hands dirty... Ilya Sutskever recommends implementing simple MNIST classifiers, small convnets, reimplementing char-rnn, and then playing with a big convnet. Personally, I started out by picking Kaggle competitions (especially the "Knowledge" ones) and using those as a source of problems. Implementing agents for OpenAI Gym (or algorithms for the set of research problems we’ll be releasing soon) could also be a good starting place.

# Letta Software Engineering Stack

- Strong proficiency with Python
- Strong understanding of how to architect services for security, reliability, and performance
- Ability to design clean, robust REST APIs
- Ability to architect robust, production-grade services
- Familiarity with IaC (Terraform) and cloud infrastructure
- Familiarity with Docker and K8
- Familiarity with tooling across the AI stack, such as inference engines (e.g. vLLM, Ollama), vector DBs (e.g. Chroma, pgvector), and RAG (e.g. llama-index, langchain)
- Bonus: proficient with TypeScript, React, Tailwind, etc. (the modern stack for web applications)

- Inference Engines
  - vLLM
  - Ollama
- vector DBs
  - Chroma
  - pgvector
- RAG
  - llama-index
  - langchain


# Layers of the Stack

- Fundamentals of Learning Theory - Linear Algebra - ee263 (not studying this too heavily)
- Model Architecture - bishop book, eecs189
- Machine learning compilation - Deep Learning Systems Course, write some pytorch
- Inference/Training Optimization - AISYS-fa2024, vllm commit
- Machine Learning Applications - Project that spans Letta Software Engineering Stack

https://huyenchip.com/ml-interviews-book/

https://github.com/emilwallner/How-to-learn-Deep-Learning
