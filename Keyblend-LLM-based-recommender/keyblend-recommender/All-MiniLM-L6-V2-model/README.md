---
base_model: sentence-transformers/all-MiniLM-L6-v2
library_name: sentence-transformers
metrics:
- pearson_cosine
- spearman_cosine
- pearson_manhattan
- spearman_manhattan
- pearson_euclidean
- spearman_euclidean
- pearson_dot
- spearman_dot
- pearson_max
- spearman_max
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- generated_from_trainer
- dataset_size:94300
- loss:CoSENTLoss
widget:
- source_sentence: corsetdress littleblackdress bodycon georgette e stylish function
    clothing design outdoor womenclothing lbd black westernwear perfect goofy datenight
    trendy travel trip night corset womenfashion date print mezenor comfort woman
    dress n
  sentences:
  - pattern neck stylish function clothing pleated design bold mannmani outdoor party
    goofy trendy plain wear travel college trip quirky semi halter soft lady office
    color print comfort girl formal woman dress tier school fashion top casual melody
    nude fabric graphic event
  - evening pattern neck half stylish function clothes clothing casuals design bold
    fashionable outdoor party goofy trendy plain wear full brunch travel college trip
    quirky fit date semi sleeve soft cool lady flare high office print smocked wine
    comfort look indian girl formal empty woman & dress school fashion style mini
    day brand dressed apparel casual fabric graphic chic event
  - meeting natural type lasting stylish function design outdoor matte party perfect
    stick glossy trendy feature wear make travel college trip long quirky lip semi
    austen finish casusal maroon lady liquid office color girl formal colour woman
    jane lipstick formulation school shaadi makeup gloss family shade event
- source_sentence: corsetdress littleblackdress bodycon georgette e stylish function
    clothing design outdoor womenclothing lbd black westernwear perfect goofy datenight
    trendy travel trip night corset womenfashion date print mezenor comfort woman
    dress n
  sentences:
  - oil shave spruce men onion healthy hair club fall red india grooming care
  - support sport designer activewear sportswear stylish gym clothes clothing fashionable
    bra yoga trendy wear run myriad cool supporting crop look girl woman parallel
    fashion style brand top apparel
  - non meeting natural type man patchy shea lasting stylish unisex cafe outdoor mango
    party stick organic glossy glow trendy feature almond pigmented make tinted travel
    college trip tanned long quirky coconut tan lip vegan body rough bright office
    extract cocoa melon berry formulation butter men balm chap makeup hazelnut gloss
    boy unrefined event
- source_sentence: corsetdress littleblackdress bodycon georgette e stylish function
    clothing design outdoor womenclothing lbd black westernwear perfect goofy datenight
    trendy travel trip night corset womenfashion date print mezenor comfort woman
    dress n
  sentences:
  - designer match mix stylish casuals gift comfortable foot trendy sock 3 oak quirky
    accessory - cool feets indian formal & mint men giftbox casual box boy
  - cutout neck cape ego party alter trendy trench wear navy v audrey sleeve girl
    a-line woman dress classy elegent
  - meeting natural type lasting stylish function design outdoor tinge agave party
    stick glossy trendy feature wear make travel college trip long quirky lip semi
    bullet finish casusal vegan lady extract cherry office color happy girl formal
    colour woman horn lipstick formulation school balm shaadi makeup dry gloss kiss,
    red family dehydrated shade event
- source_sentence: corsetdress littleblackdress bodycon georgette e stylish function
    clothing design outdoor womenclothing lbd black westernwear perfect goofy datenight
    trendy travel trip night corset womenfashion date print mezenor comfort woman
    dress n
  sentences:
  - pattern half stylish function clothing 15 design bold button outdoor party goofy
    cotton trendy plain wear navy collared travel college trip quirky contrast semi
    soft sleeve collar lady office print crop comfort shirt girl formal woman blue
    dress school top casual fabric graphic event
  - pattern white stylish function clothing design bold outdoor striped black party
    goofy trendy plain wear travel college trip quirky semi soft lady office print
    cami mezenor comfort girl formal woman dress school top casual fabric graphic
    event
  - meeting type pattern neck half stylish function clothing casuals design outdoor
    black waist party organic cotton trendy length wear full travel college trip quirky
    tone two floral semi soft kurta lady office color comfort indian girl formal colour
    woman empty dress school comforting shaadi brand casual apparel printed ethnic
    fabric event work
- source_sentence: corsetdress littleblackdress bodycon georgette e stylish function
    clothing design outdoor womenclothing lbd black westernwear perfect goofy datenight
    trendy travel trip night corset womenfashion date print mezenor comfort woman
    dress n
  sentences:
  - pattern stylish function clothing 15 design bold button outdoor party goofy cotton
    trendy plain wear pistachio collared cross travel college tie trip quirky semi
    soft collar lady office print tie-down tie- comfort shirt girl formal woman dress
    school botton top casual green fabric graphic event
  - anti oil almond men shaving hair castor lavender dryness fall bombay company coconut
    care strong woman advanced
  - meeting natural type lasting devoted function stylish design outdoor tinge matte
    party stick glossy trendy feature wear make devote travel college trip long quirky
    lip semi finish casusal lady liquid office color girl formal colour woman lipstick
    pink formulation school shaadi makeup gloss red nude family shade event
model-index:
- name: SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2
  results:
  - task:
      type: semantic-similarity
      name: Semantic Similarity
    dataset:
      name: score
      type: score
    metrics:
    - type: pearson_cosine
      value: 0.7920942334173211
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.7481064262842818
      name: Spearman Cosine
    - type: pearson_manhattan
      value: 0.5217120825326587
      name: Pearson Manhattan
    - type: spearman_manhattan
      value: 0.4847839635546036
      name: Spearman Manhattan
    - type: pearson_euclidean
      value: 0.5209394481211068
      name: Pearson Euclidean
    - type: spearman_euclidean
      value: 0.48363715971010235
      name: Spearman Euclidean
    - type: pearson_dot
      value: 0.38576821082133256
      name: Pearson Dot
    - type: spearman_dot
      value: 0.3673432069235592
      name: Spearman Dot
    - type: pearson_max
      value: 0.7920942334173211
      name: Pearson Max
    - type: spearman_max
      value: 0.7481064262842818
      name: Spearman Max
    - type: pearson_cosine
      value: 0.8484699578100623
      name: Pearson Cosine
    - type: spearman_cosine
      value: 0.8167519904433979
      name: Spearman Cosine
    - type: pearson_manhattan
      value: 0.6400774914935498
      name: Pearson Manhattan
    - type: spearman_manhattan
      value: 0.608351974172603
      name: Spearman Manhattan
    - type: pearson_euclidean
      value: 0.6416515121850785
      name: Pearson Euclidean
    - type: spearman_euclidean
      value: 0.6090899294059726
      name: Spearman Euclidean
    - type: pearson_dot
      value: 0.5325063200161753
      name: Pearson Dot
    - type: spearman_dot
      value: 0.5062764423591809
      name: Spearman Dot
    - type: pearson_max
      value: 0.8484699578100623
      name: Pearson Max
    - type: spearman_max
      value: 0.8167519904433979
      name: Spearman Max
---

# SentenceTransformer based on sentence-transformers/all-MiniLM-L6-v2

This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) on the csv dataset. It maps sentences & paragraphs to a 384-dimensional dense vector space and can be used for semantic textual similarity, semantic search, paraphrase mining, text classification, clustering, and more.

## Model Details

### Model Description
- **Model Type:** Sentence Transformer
- **Base model:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) <!-- at revision 8b3219a92973c328a8e22fadcfa821b5dc75636a -->
- **Maximum Sequence Length:** 512 tokens
- **Output Dimensionality:** 384 tokens
- **Similarity Function:** Cosine Similarity
- **Training Dataset:**
    - csv
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/UKPLab/sentence-transformers)
- **Hugging Face:** [Sentence Transformers on Hugging Face](https://huggingface.co/models?library=sentence-transformers)

### Full Model Architecture

```
SentenceTransformer(
  (0): Transformer({'max_seq_length': 512, 'do_lower_case': False}) with Transformer model: BertModel 
  (1): Pooling({'word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False, 'pooling_mode_weightedmean_tokens': False, 'pooling_mode_lasttoken': False, 'include_prompt': True})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import SentenceTransformer

# Download from the 🤗 Hub
model = SentenceTransformer("sentence_transformers_model_id")
# Run inference
sentences = [
    'corsetdress littleblackdress bodycon georgette e stylish function clothing design outdoor womenclothing lbd black westernwear perfect goofy datenight trendy travel trip night corset womenfashion date print mezenor comfort woman dress n',
    'anti oil almond men shaving hair castor lavender dryness fall bombay company coconut care strong woman advanced',
    'pattern stylish function clothing 15 design bold button outdoor party goofy cotton trendy plain wear pistachio collared cross travel college tie trip quirky semi soft collar lady office print tie-down tie- comfort shirt girl formal woman dress school botton top casual green fabric graphic event',
]
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# Get the similarity scores for the embeddings
similarities = model.similarity(embeddings, embeddings)
print(similarities.shape)
# [3, 3]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

## Evaluation

### Metrics

#### Semantic Similarity
* Dataset: `score`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| pearson_cosine      | 0.7921     |
| **spearman_cosine** | **0.7481** |
| pearson_manhattan   | 0.5217     |
| spearman_manhattan  | 0.4848     |
| pearson_euclidean   | 0.5209     |
| spearman_euclidean  | 0.4836     |
| pearson_dot         | 0.3858     |
| spearman_dot        | 0.3673     |
| pearson_max         | 0.7921     |
| spearman_max        | 0.7481     |

#### Semantic Similarity
* Dataset: `score`
* Evaluated with [<code>EmbeddingSimilarityEvaluator</code>](https://sbert.net/docs/package_reference/sentence_transformer/evaluation.html#sentence_transformers.evaluation.EmbeddingSimilarityEvaluator)

| Metric              | Value      |
|:--------------------|:-----------|
| pearson_cosine      | 0.8485     |
| **spearman_cosine** | **0.8168** |
| pearson_manhattan   | 0.6401     |
| spearman_manhattan  | 0.6084     |
| pearson_euclidean   | 0.6417     |
| spearman_euclidean  | 0.6091     |
| pearson_dot         | 0.5325     |
| spearman_dot        | 0.5063     |
| pearson_max         | 0.8485     |
| spearman_max        | 0.8168     |

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### csv

* Dataset: csv
* Size: 94,300 training samples
* Columns: <code>sentence1</code>, <code>sentence2</code>, and <code>score</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence1                                                                         | sentence2                                                                           | score                                                          |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                              | float                                                          |
  | details | <ul><li>min: 54 tokens</li><li>mean: 54.0 tokens</li><li>max: 54 tokens</li></ul> | <ul><li>min: 12 tokens</li><li>mean: 49.44 tokens</li><li>max: 116 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.49</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence1                                                                                                                                                                                                                                                                                                      | sentence2                                                                                                                                                                                                                                                                                                             | score             |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|
  | <code>skincare meeting natural tested type cosmetic protect correctly benefit lasting line radiance use outdoor party damaged feature face combo pimple travel vera long even tone semi soft vegan duo wrinkle free formal firm school serum product cheek makeup combination aloe gloss facecare event</code> | <code>dress western trendy wear ruffle shreetatvam indian india peach woman</code>                                                                                                                                                                                                                                    | <code>0.52</code> |
  | <code>skincare meeting natural tested type cosmetic protect correctly benefit lasting line radiance use outdoor party damaged feature face combo pimple travel vera long even tone semi soft vegan duo wrinkle free formal firm school serum product cheek makeup combination aloe gloss facecare event</code> | <code>designer protect meow sassy beige function stylish design outdoor occasion cat comfortable party cotton bacteria trendy face mask comfy fancy quirky saftey protection lady print virus everyday girl lightweight woman doodle style resuable shaadi reversible embroidery printed graphic sequoia event</code> | <code>0.55</code> |
  | <code>skincare meeting natural tested type cosmetic protect correctly benefit lasting line radiance use outdoor party damaged feature face combo pimple travel vera long even tone semi soft vegan duo wrinkle free formal firm school serum product cheek makeup combination aloe gloss facecare event</code> | <code>designer urban grey funky trendy cool sneaker men stylish shoe canvas printed anime boy quirky footwear foot pitara</code>                                                                                                                                                                                      | <code>0.3</code>  |
* Loss: [<code>CoSENTLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosentloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "pairwise_cos_sim"
  }
  ```

### Evaluation Dataset

#### csv

* Dataset: csv
* Size: 56,580 evaluation samples
* Columns: <code>sentence1</code>, <code>sentence2</code>, and <code>score</code>
* Approximate statistics based on the first 1000 samples:
  |         | sentence1                                                                         | sentence2                                                                           | score                                                          |
  |:--------|:----------------------------------------------------------------------------------|:------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type    | string                                                                            | string                                                                              | float                                                          |
  | details | <ul><li>min: 57 tokens</li><li>mean: 57.0 tokens</li><li>max: 57 tokens</li></ul> | <ul><li>min: 12 tokens</li><li>mean: 48.73 tokens</li><li>max: 116 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.54</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence1                                                                                                                                                                                                                                                | sentence2                                                                                                                                                                                                                                                      | score             |
  |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------|
  | <code>corsetdress littleblackdress bodycon georgette e stylish function clothing design outdoor womenclothing lbd black westernwear perfect goofy datenight trendy travel trip night corset womenfashion date print mezenor comfort woman dress n</code> | <code>mumbai stylish comfortable baseball embroidered womenswear female city fit snapback 1947ind retro comfort menswear india woman male mum men cap casual embroidery red 1947</code>                                                                        | <code>0.18</code> |
  | <code>corsetdress littleblackdress bodycon georgette e stylish function clothing design outdoor womenclothing lbd black westernwear perfect goofy datenight trendy travel trip night corset womenfashion date print mezenor comfort woman dress n</code> | <code>white pattern stylish design rivir outdoor waist party trendy plain length gent shoe college rubber lace travel quirky trip fit soft sneaker high office color comfort footwear colour blue school men rise men's casual fabric boy graphic event</code> | <code>0.44</code> |
  | <code>corsetdress littleblackdress bodycon georgette e stylish function clothing design outdoor womenclothing lbd black westernwear perfect goofy datenight trendy travel trip night corset womenfashion date print mezenor comfort woman dress n</code> | <code>sling travelling luggage set laptop travel backpack cabin batua 4 kit bag voyager office leather girl toiletry woman duffle men green boy native</code>                                                                                                  | <code>0.28</code> |
* Loss: [<code>CoSENTLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#cosentloss) with these parameters:
  ```json
  {
      "scale": 20.0,
      "similarity_fct": "pairwise_cos_sim"
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `eval_strategy`: steps
- `learning_rate`: 2e-05
- `num_train_epochs`: 5
- `warmup_ratio`: 0.1
- `save_only_model`: True
- `seed`: 33
- `fp16`: True
- `load_best_model_at_end`: True

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `overwrite_output_dir`: False
- `do_predict`: False
- `eval_strategy`: steps
- `prediction_loss_only`: True
- `per_device_train_batch_size`: 8
- `per_device_eval_batch_size`: 8
- `per_gpu_train_batch_size`: None
- `per_gpu_eval_batch_size`: None
- `gradient_accumulation_steps`: 1
- `eval_accumulation_steps`: None
- `torch_empty_cache_steps`: None
- `learning_rate`: 2e-05
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `max_grad_norm`: 1.0
- `num_train_epochs`: 5
- `max_steps`: -1
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: {}
- `warmup_ratio`: 0.1
- `warmup_steps`: 0
- `log_level`: passive
- `log_level_replica`: warning
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `save_safetensors`: True
- `save_on_each_node`: False
- `save_only_model`: True
- `restore_callback_states_from_checkpoint`: False
- `no_cuda`: False
- `use_cpu`: False
- `use_mps_device`: False
- `seed`: 33
- `data_seed`: None
- `jit_mode_eval`: False
- `use_ipex`: False
- `bf16`: False
- `fp16`: True
- `fp16_opt_level`: O1
- `half_precision_backend`: auto
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `local_rank`: 0
- `ddp_backend`: None
- `tpu_num_cores`: None
- `tpu_metrics_debug`: False
- `debug`: []
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_prefetch_factor`: None
- `past_index`: -1
- `disable_tqdm`: False
- `remove_unused_columns`: True
- `label_names`: None
- `load_best_model_at_end`: True
- `ignore_data_skip`: False
- `fsdp`: []
- `fsdp_min_num_params`: 0
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `fsdp_transformer_layer_cls_to_wrap`: None
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `deepspeed`: None
- `label_smoothing_factor`: 0.0
- `optim`: adamw_torch
- `optim_args`: None
- `adafactor`: False
- `group_by_length`: False
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `skip_memory_metrics`: True
- `use_legacy_prediction_loop`: False
- `push_to_hub`: False
- `resume_from_checkpoint`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_private_repo`: False
- `hub_always_push`: False
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `include_inputs_for_metrics`: False
- `eval_do_concat_batches`: True
- `fp16_backend`: auto
- `push_to_hub_model_id`: None
- `push_to_hub_organization`: None
- `mp_parameters`: 
- `auto_find_batch_size`: False
- `full_determinism`: False
- `torchdynamo`: None
- `ray_scope`: last
- `ddp_timeout`: 1800
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `dispatch_batches`: None
- `split_batches`: None
- `include_tokens_per_second`: False
- `include_num_input_tokens_seen`: False
- `neftune_noise_alpha`: None
- `optim_target_modules`: None
- `batch_eval_metrics`: False
- `eval_on_start`: False
- `eval_use_gather_object`: False
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional

</details>

### Training Logs
<details><summary>Click to expand</summary>

| Epoch      | Step     | Training Loss | loss       | score_spearman_cosine |
|:----------:|:--------:|:-------------:|:----------:|:---------------------:|
| 0.0424     | 500      | 5.0148        | 3.6542     | 0.2169                |
| 0.0848     | 1000     | 3.2089        | 3.3720     | 0.4387                |
| 0.1272     | 1500     | 3.1511        | 3.3683     | 0.4619                |
| 0.1697     | 2000     | 3.1123        | 3.3377     | 0.5250                |
| 0.2121     | 2500     | 3.0766        | 3.2805     | 0.5861                |
| 0.2545     | 3000     | 3.02          | 3.2505     | 0.6066                |
| 0.2969     | 3500     | 2.9882        | 3.2296     | 0.6379                |
| 0.3393     | 4000     | 2.972         | 3.2570     | 0.6496                |
| 0.3817     | 4500     | 2.9579        | 3.3504     | 0.6554                |
| 0.4242     | 5000     | 2.9247        | 3.2242     | 0.6663                |
| 0.4666     | 5500     | 2.936         | 3.1538     | 0.6844                |
| 0.5090     | 6000     | 2.8915        | 3.2049     | 0.6838                |
| 0.5514     | 6500     | 2.8988        | 3.1907     | 0.6944                |
| **0.5938** | **7000** | **2.872**     | **3.1581** | **0.7013**            |
| 0.6362     | 7500     | 2.8162        | 3.2054     | 0.7114                |
| 0.6787     | 8000     | 2.8301        | 3.1793     | 0.7173                |
| 0.7211     | 8500     | 2.8244        | 3.1795     | 0.7284                |
| 0.7635     | 9000     | 2.8115        | 3.1521     | 0.7346                |
| 0.8059     | 9500     | 2.7862        | 3.1605     | 0.7345                |
| 0.8483     | 10000    | 2.8028        | 3.1384     | 0.7439                |
| 0.8907     | 10500    | 2.7844        | 3.1539     | 0.7465                |
| 0.9332     | 11000    | 2.7617        | 3.1648     | 0.7450                |
| 0.9756     | 11500    | 2.7895        | 3.1423     | 0.7481                |
| 0.0424     | 500      | 3.1995        | -          | -                     |
| 0.0848     | 1000     | 3.2511        | -          | -                     |
| 0.1272     | 1500     | 3.1972        | -          | -                     |
| 0.1697     | 2000     | 3.1582        | -          | -                     |
| 0.2121     | 2500     | 3.1366        | -          | -                     |
| 0.2545     | 3000     | 3.0848        | -          | -                     |
| 0.2969     | 3500     | 3.0675        | -          | -                     |
| 0.3393     | 4000     | 3.0544        | -          | -                     |
| 0.3817     | 4500     | 3.0429        | -          | -                     |
| 0.4242     | 5000     | 2.9998        | -          | -                     |
| 0.4666     | 5500     | 3.0159        | -          | -                     |
| 0.5090     | 6000     | 2.9819        | -          | -                     |
| 0.5514     | 6500     | 2.9849        | -          | -                     |
| **0.5938** | **7000** | **2.9428**    | **3.1886** | **0.6678**            |
| 0.6362     | 7500     | 2.9103        | -          | -                     |
| 0.6787     | 8000     | 2.9003        | -          | -                     |
| 0.7211     | 8500     | 2.9077        | -          | -                     |
| 0.7635     | 9000     | 2.8907        | -          | -                     |
| 0.8059     | 9500     | 2.846         | -          | -                     |
| 0.8483     | 10000    | 2.865         | -          | -                     |
| 0.8907     | 10500    | 2.8488        | -          | -                     |
| 0.9332     | 11000    | 2.841         | -          | -                     |
| 0.9756     | 11500    | 2.8407        | -          | -                     |
| 1.0180     | 12000    | 2.8322        | -          | -                     |
| 1.0604     | 12500    | 2.7742        | -          | -                     |
| 1.1028     | 13000    | 2.7889        | -          | -                     |
| 1.1452     | 13500    | 2.7799        | -          | -                     |
| 1.1876     | 14000    | 2.7648        | 3.0949     | 0.7542                |
| 1.2301     | 14500    | 2.7786        | -          | -                     |
| 1.2725     | 15000    | 2.7818        | -          | -                     |
| 1.3149     | 15500    | 2.7462        | -          | -                     |
| 1.3573     | 16000    | 2.7794        | -          | -                     |
| 1.3997     | 16500    | 2.7419        | -          | -                     |
| 1.4421     | 17000    | 2.7436        | -          | -                     |
| 1.4846     | 17500    | 2.7276        | -          | -                     |
| 1.5270     | 18000    | 2.7192        | -          | -                     |
| 1.5694     | 18500    | 2.7619        | -          | -                     |
| 1.6118     | 19000    | 2.752         | -          | -                     |
| 1.6542     | 19500    | 2.7085        | -          | -                     |
| 1.6966     | 20000    | 2.6882        | -          | -                     |
| 1.7391     | 20500    | 2.7291        | -          | -                     |
| 1.7815     | 21000    | 2.7144        | 3.1244     | 0.7818                |
| 1.8239     | 21500    | 2.7162        | -          | -                     |
| 1.8663     | 22000    | 2.711         | -          | -                     |
| 1.9087     | 22500    | 2.7024        | -          | -                     |
| 1.9511     | 23000    | 2.7253        | -          | -                     |
| 1.9936     | 23500    | 2.7057        | -          | -                     |
| 2.0360     | 24000    | 2.6631        | -          | -                     |
| 2.0784     | 24500    | 2.6362        | -          | -                     |
| 2.1208     | 25000    | 2.652         | -          | -                     |
| 2.1632     | 25500    | 2.6203        | -          | -                     |
| 2.2056     | 26000    | 2.6576        | -          | -                     |
| 2.2480     | 26500    | 2.6575        | -          | -                     |
| 2.2905     | 27000    | 2.626         | -          | -                     |
| 2.3329     | 27500    | 2.648         | -          | -                     |
| 2.3753     | 28000    | 2.6612        | 3.0779     | 0.7921                |
| 2.4177     | 28500    | 2.637         | -          | -                     |
| 2.4601     | 29000    | 2.6505        | -          | -                     |
| 2.5025     | 29500    | 2.603         | -          | -                     |
| 2.5450     | 30000    | 2.6398        | -          | -                     |
| 2.5874     | 30500    | 2.6338        | -          | -                     |
| 2.6298     | 31000    | 2.6252        | -          | -                     |
| 2.6722     | 31500    | 2.6291        | -          | -                     |
| 2.7146     | 32000    | 2.6255        | -          | -                     |
| 2.7570     | 32500    | 2.6338        | -          | -                     |
| 2.7995     | 33000    | 2.6231        | -          | -                     |
| 2.8419     | 33500    | 2.6171        | -          | -                     |
| 2.8843     | 34000    | 2.6257        | -          | -                     |
| 2.9267     | 34500    | 2.5972        | -          | -                     |
| 2.9691     | 35000    | 2.6093        | 3.0821     | 0.7935                |
| 3.0115     | 35500    | 2.6277        | -          | -                     |
| 3.0540     | 36000    | 2.6033        | -          | -                     |
| 3.0964     | 36500    | 2.5664        | -          | -                     |
| 3.1388     | 37000    | 2.5604        | -          | -                     |
| 3.1812     | 37500    | 2.6064        | -          | -                     |
| 3.2236     | 38000    | 2.5646        | -          | -                     |
| 3.2660     | 38500    | 2.5515        | -          | -                     |
| 3.3084     | 39000    | 2.5509        | -          | -                     |
| 3.3509     | 39500    | 2.5612        | -          | -                     |
| 3.3933     | 40000    | 2.5126        | -          | -                     |
| 3.4357     | 40500    | 2.5611        | -          | -                     |
| 3.4781     | 41000    | 2.5553        | -          | -                     |
| 3.5205     | 41500    | 2.547         | -          | -                     |
| 3.5629     | 42000    | 2.5735        | 3.1884     | 0.8013                |
| 3.6054     | 42500    | 2.5142        | -          | -                     |
| 3.6478     | 43000    | 2.5417        | -          | -                     |
| 3.6902     | 43500    | 2.5371        | -          | -                     |
| 3.7326     | 44000    | 2.5479        | -          | -                     |
| 3.7750     | 44500    | 2.4711        | -          | -                     |
| 3.8174     | 45000    | 2.5424        | -          | -                     |
| 3.8599     | 45500    | 2.5162        | -          | -                     |
| 3.9023     | 46000    | 2.5325        | -          | -                     |
| 3.9447     | 46500    | 2.5736        | -          | -                     |
| 3.9871     | 47000    | 2.5622        | -          | -                     |
| 4.0295     | 47500    | 2.4703        | -          | -                     |
| 4.0719     | 48000    | 2.4603        | -          | -                     |
| 4.1144     | 48500    | 2.4935        | -          | -                     |
| 4.1568     | 49000    | 2.5569        | 3.1473     | 0.8126                |
| 4.1992     | 49500    | 2.4939        | -          | -                     |
| 4.2416     | 50000    | 2.4899        | -          | -                     |
| 4.2840     | 50500    | 2.4666        | -          | -                     |
| 4.3264     | 51000    | 2.4807        | -          | -                     |
| 4.3688     | 51500    | 2.4741        | -          | -                     |
| 4.4113     | 52000    | 2.4452        | -          | -                     |
| 4.4537     | 52500    | 2.4739        | -          | -                     |
| 4.4961     | 53000    | 2.497         | -          | -                     |
| 4.5385     | 53500    | 2.5168        | -          | -                     |
| 4.5809     | 54000    | 2.4784        | -          | -                     |
| 4.6233     | 54500    | 2.47          | -          | -                     |
| 4.6658     | 55000    | 2.4589        | -          | -                     |
| 4.7082     | 55500    | 2.4769        | -          | -                     |
| 4.7506     | 56000    | 2.4715        | 3.1551     | 0.8168                |
| 4.7930     | 56500    | 2.4943        | -          | -                     |
| 4.8354     | 57000    | 2.5042        | -          | -                     |
| 4.8778     | 57500    | 2.468         | -          | -                     |
| 4.9203     | 58000    | 2.4878        | -          | -                     |
| 4.9627     | 58500    | 2.5167        | -          | -                     |

* The bold row denotes the saved checkpoint.
</details>

### Framework Versions
- Python: 3.10.14
- Sentence Transformers: 3.1.0
- Transformers: 4.44.0
- PyTorch: 2.4.0
- Accelerate: 0.33.0
- Datasets: 2.21.0
- Tokenizers: 0.19.1

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

#### CoSENTLoss
```bibtex
@online{kexuefm-8847,
    title={CoSENT: A more efficient sentence vector scheme than Sentence-BERT},
    author={Su Jianlin},
    year={2022},
    month={Jan},
    url={https://kexue.fm/archives/8847},
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->