# AboutMe: Using Self-Descriptions in Webpages to Document the Effects of English Pretraining Data Filters

## Paper

**Authors**: Li Lucy, Suchin Gururangan, Luca Soldaini, Emma Strubell, David Bamman, Lauren Klein, Jesse Dodge

**Abstract**: Large language models' (LLMs) abilities are drawn from their pretraining data, and model development begins with data curation. However, decisions around what data is retained or removed during this initial stage is under-scrutinized. In our work, we ground web text, which is a popular pretraining data source, to its social and geographic contexts. We create a new dataset of 10.3 million self-descriptions of website creators, and extract information about who they are and where they are from: their topical interests, social roles, and geographic affiliations. Then, we conduct the first study investigating how ten "quality" and English language identification (langID) filters affect webpages that vary along these social dimensions. Our experiments illuminate a range of implicit preferences in data curation: we show that some quality classifiers act like topical domain filters, and langID can overlook English content from some regions of the world. Overall, we hope that our work will encourage a new line of research on pretraining data curation practices and its social implications.

[Link](https://lucy3.github.io/preprint.pdf)

## Code Directory

**Code**
- **cluster**
  - `cluster.py`
  - `train_clusterer.py`
- **filter**
  - **lr**
     - `hyperparameters.py`
     - `lr_quality_filters.py`
     - `train.py`
     - `util.py`
  - `evaluate_ft_models.py`
  - `quality_data_org.py`
  - `rule_based_scores.py`
  - `sample_openwebtext2.py`
  - `score_manager.py`
  - `text_normalizer.py`
  - `wikipedia_perplexity.py`
  - `zreader.py`
- **get\_data**
  - `bloomfilter.py`
  - `dataset_statistics.py`
  - `get_random_pages.py`
  - `url_processor.py`
  - `website_expander.py`
- **identity\_measures**
  - **geography**
  - **personas**
  - **roberta\_classifier**
  - `person_vs_orgs.py`
  - `spacy_helper.py`

## Dataset 

Link to huggingface dataset coming soon 
