# Olfactory Bulb Detection

This project is based on the research paper:

**"Dense and persistent odor representations in the olfactory bulb of awake mice"**

*Delaram Pirhayati, Cameron L Smith, Ryan Kroeger, Saket Navlakha, Paul Pfaffinger, Jacob Reimer, Benjamin R Arenkiel, Ankit Patel, Elizabeth H Moss*

Journal of Neuroscience, 2024, Vol. 44, Issue 39

**Paper Link:** https://www.jneurosci.org/content/44/39/e0116242024.abstract

DOI: 10.1523/JNEUROSCI.0116-24.2024

## Project Overview

OB_Detection is a Python project for analyzing olfactory bulb (OB) data, focusing on investigating dense and persistent odor representations using machine learning techniques. This project implements the data analysis methods described in our paper, including:

1. Processing of meso-scale two-photon calcium imaging data from the mouse olfactory bulb
2. Analysis of odor-evoked glomerular activity patterns
3. Machine learning classifiers for odor discrimination
4. Value-added analyses to investigate the density and redundancy of odor representations
5. Random Forest feature selection using Gini Impurity for glomerular ranking
6. Temporal analysis of odor information persistence

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/OB_Detection.git
   cd OB_Detection
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main analysis pipeline:

```
python main.py
```

This script executes the entire analysis workflow, including data preprocessing, classifier training, value-added analyses, and visualization of results.

## Key Features

- Preprocessing of two-photon calcium imaging data
- Implementation of various classifiers
- Value-added analyses for investigating odor information distribution
- Random Forest feature importance calculation using Gini Impurity
- Temporal analysis of odor information persistence
- Visualization tools for results and statistical analyses

## Citation

If you use this code in your research, please cite our paper:

Pirhayati, D., Smith, C. L., Kroeger, R., Navlakha, S., Pfaffinger, P., Reimer, J., Arenkiel, B. R., Patel, A., & Moss, E. H. (2024). Dense and persistent odor representations in the olfactory bulb of awake mice. Journal of Neuroscience, 44(39), e0116242024. https://doi.org/10.1523/JNEUROSCI.0116-24.2024

## License

This project is licensed under the MIT License.

## Contact

For questions or feedback, please contact dp43@rice.edu.
