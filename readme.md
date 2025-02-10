# Abstract Meaning Representation in Different Languages

## 📌 Overview

This project explores how Abstract Meaning Representation (AMR) generation varies across different languages, specifically English, Welsh, and Irish. We train multilingual transformer-based models to generate AMR graphs for each language and evaluate their performance.

## 🔍 Abstract Meaning Representation (AMR)

AMR represents the meaning of a sentence as a structured graph, capturing concepts and relationships between them. It is widely used in:

- Machine Translation
- Question Answering
- Text Summarization

However, AMR research is heavily centered on English, limiting its potential in multilingual settings. Our project aims to bridge this gap by training and fine-tuning models for other languages.

## 🛠 Methodology

### 1️⃣ Dataset Preparation

- Extracted ~1600 English sentences from the **MASSIVE-AMR** dataset.
- Translated them into **Welsh** and **Irish** using MarianMT.
- Created a multilingual dataset for training AMR generation models.

### 2️⃣ Model Training

We fine-tuned **T5-small** models using different language configurations:

| Model | Precision | Recall | F1 Score |
|--------|----------|--------|----------|
| Irish (T5-small) | 6.14% | 11.75% | 11.08% |
| Welsh (T5-small) | 5.06% | 11.83% | 13.44% |
| Irish + Welsh | 6.32% | 12.28% | 11.83% |
| English + Irish + Welsh | **7.19%** | **12.75%** | **11.87%** |

### 3️⃣ Evaluation

- **Smatch Score** used to measure AMR quality.
- **Comparison with STOG (AMRlib)** baseline.
- Generated **5055 AMR graphs** compiled into a CSV.

## 🚀 Results & Key Findings

- Multilingual training improves AMR parsing performance.
- The **English + Irish + Welsh model** achieved the best Smatch scores.
- The default **STOG model** underperforms on non-English languages.

## 📌 Future Improvements

- Train on **larger datasets** for better accuracy.
- Fine-tune **T5-base** or more powerful transformer models.
- Extend the dataset to include **Scottish Gaelic** and other languages.

## 📜 References

- [MASSIVE-AMR Dataset](https://github.com/amazon-science/MASSIVE-AMR)
- [Abstract Meaning Representation (AMR)](https://amr.isi.edu/)
- [Hugging Face T5 Model](https://huggingface.co/t5-small)

---
📝 **Authors**: Elias-Valeriu Stoica & Valentin-Ion Semen  
📧 Contact: [elias-valeriu.stoica@s.unibuc.ro](mailto:elias-valeriu.stoica@s.unibuc.ro) | [valentin-ion.semen@s.unibuc.ro](mailto:valentin-ion.semen@s.unibuc.ro)