# Experiential Inequality in Social Infrastructure Quality

![Python](https://img.shields.io/badge/python-3.10%2B-blue)  
![Status](https://img.shields.io/badge/status-active-success)  
![License](https://img.shields.io/badge/license-MIT-green)

This repository provides a **scalable computational framework** for measuring *experiential inequality* in social infrastructure (SI) quality using millions of crowdsourced reviews.  

It accompanies the paper:  
**“Beyond Access: Lived-Experience Narratives Reveal Hidden Inequalities in Social Infrastructure Quality”**

---

## 🌍 Why this matters

Conventional infrastructure metrics focus on *availability* or *proximity*. These measures assume that access implies benefit.

This project shows that:

- Infrastructure can be **present but systematically low quality**
- Quality is **unevenly experienced across socioeconomic groups**
- These disparities persist even after controlling for geography

We therefore introduce:

> **Experiential inequality**: unequal exposure to the *quality* of infrastructure, not just its presence.

---

## 🧠 Conceptual Pipeline

```
Places → Reviews → ABSA → Quality Index → Exposure → Inequality
```

1. Extract infrastructure locations  
2. Collect and process user reviews  
3. Apply multi-agent ABSA  
4. Construct place-level quality scores  
5. Aggregate exposure at SA2 level  
6. Model structural inequality  

---

## 🗂️ Repository Structure

```
1. Place extraction/
2. Review_scraping/
3. Review_parsing/
4. Sample_annotation/
5. Agent_validation/
6. ABSA_results/
7. ABSA_Analysis/
8. Wellbeing analysis/
data/
```

### 1. Place extraction  
Retrieves social infrastructure locations using APIs (e.g., Google Places).  
Outputs structured place-level datasets.

### 2. Review_scraping  
Collects large-scale user reviews with batching and rate-limit handling.  

### 3. Review_parsing  
Cleans and structures text for downstream modeling:
- Filtering  
- Tokenization  
- Formatting for ABSA  

### 4. Sample_annotation  
Human-labeled dataset for validation and benchmarking.  

### 5. Agent_validation  
Design and evaluation of the multi-agent ABSA system, including:
- Prompt engineering  
- Supervisor validation logic  
- Consistency checks  

### 6. ABSA_results  
Outputs from ABSA:
- Aspect terms  
- Opinion terms  
- Sentiment labels  

### 7. ABSA_Analysis  
Transforms ABSA outputs into quality metrics:
- Aspect aggregation  
- Weighting (review-weighted / TF-IDF inspired)  
- Place-level quality index  

### 8. Wellbeing analysis  
Final analytical layer:
- Population-weighted exposure  
- Inequality metrics  
- Regression models (with controls and fixed effects)  

### data/  
Intermediate and processed datasets (subject to sharing constraints).

---

## ⚙️ Methodological Design

### Multi-agent ABSA framework
- Aspect Term Extraction (ATE)  
- Opinion Term Extraction (OTE)  
- Aspect-Level Sentiment Classification (ALSC)  
- Supervisor agent for validation and conflict resolution  

### Quality index construction
- Sentiment aggregated at aspect level  
- Weighted by engagement (review counts / TF-IDF)  
- Produces a composite place-level quality score  

### Exposure modeling
- Aggregation to SA2 using population-weighted logic  
- Captures how people are exposed to distributions of quality  

---

## 📊 Outputs

The pipeline produces:

- Place-level quality indices  
- SA2-level exposure measures  
- Inequality metrics (distributional and group-based)  
- Regression-ready datasets  

---

## 🔁 Reproducibility Guide

Run the pipeline sequentially:

```bash
# 1. Extract places
# 2. Scrape reviews
# 3. Parse and clean text
# 4. Run ABSA pipeline
# 5. Construct quality index
# 6. Aggregate to SA2
# 7. Run statistical models
```

Each module is independent and can be adapted or reused.

---

## 📦 Requirements

- Python ≥ 3.10  
- pandas, numpy, geopandas  
- scikit-learn, statsmodels  
- DSPy / LangGraph (for agent orchestration)  
- API access (e.g., Google Places)

---

## ⚠️ Data Availability

Due to platform restrictions:

- Raw review data may not be redistributed  
- Processed data and code are provided for reproducibility  
- Users may need to re-run scraping components with API access  

---

## 📌 Key Contribution

This repository introduces a new measurement paradigm:

- Moves beyond access-based metrics  
- Captures lived experience at scale  
- Reveals hidden inequalities in infrastructure systems  
- Provides a scalable framework for SDG-relevant monitoring  

---

## 📖 Citation

If you use this repository, please cite:

```
Beyond Access: Lived-Experience Narratives Reveal Hidden Inequalities in Social Infrastructure Quality
```

---

## 🤝 Acknowledgements

This work integrates advances in:
- Crowdsourced data analysis  
- Natural language processing  
- Multi-agent AI systems  
- Spatial inequality modeling  

---

## 📬 Contact

For questions, collaborations, or data access:  
[Add your email or GitHub profile]
