# KeyBlend Recommender : a Product Recommendation Engine for matching contents with products 🤖

## Problem statement :

This project is about creating a product recommendation engine for a company named HYPD for their creators that will automatically suggest relevant products to tag when they upload a pebble/video content.

## Who are the creators?

Creators are core to the HYPD ecosystem. Creators are often influencers or individuals
who can create shops on HYPD and share content to sell the products (their own or
other brands) to their audience.

## What content do creators create?

Creators can upload pebbles on HYPD, which are nothing but video content, the pebble has a video and product associated with it.

## Data Description :

It consists of two datasets - Catalog and Content descriptions.

**Catalog data description**

* id: Unique identifier of the catalog
* name: Name of product
* brand_id: ID of product brand
* keywords: Keywords associated with product
* retail_price: Retail price of product
* base_price: Base price of product
* cat_one: Top-level Category ID (Eg: Men)
* cat_two: 2nd Level Category ID (Eg: Clothing)
* cat_three: Third level Category ID (Eg: Shirts)
* status: Publish or Unpublished

**Content data description**

* id: Unique identifier of the content (content_id)
* type: Content type
* media_type: Format of content
* influencer_ids: Unique identifier of influencers
* brand_ids: Unique identifier of brands
* label: Labels associated with the content like Interests, Gender, etc.
* is_processed: Yes or No
* is_active: Yes or No
* view_count: View count of the content
* like_count: Like count on the content
* comment_count: Comment count on the content
* caption: Caption of the content
* catalog_ids: Identifier of catalogs
* catalog_info: Information of catalog
* created_at: timestamp of created
* processed_at: timestamp of created
* like_ids: ids of likes
* liked_by: ids of users who like the content
* last_sync: timestamp of content last synced
* category_path: path of category
* hashtags: hashtags of content

## Idea of the Project

The idea is to fine tune a Sentence-Transformer LLM that analyses the caption, hashtags and interests that come with each content created to try and predict the relevent products that could be related to the content.

## Files and folders' descriptions :  

**Data_preparation.ipynb** : It's the notebook that contains the data creating and preprocessing process.

**fine-tuning-the-llm-for-keyblend-recommender.ipynb** : It's the notebook that contains the LLM fine tuning process.

**keyblend.py** : It's the python file that contains the class implementation for the keyblend-recommender.

**main.py** : It's the python file that contains the code for implementation of the API that allows for easy communication with the recommender.

**Data** : Contains the data used for this project.

**All-MiniLM-L6-V2-model** : Contains the files of the fine tuned LLM.

**keyblend-recommender** : Contains the files for the api application.


