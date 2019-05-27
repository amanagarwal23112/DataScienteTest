# DataScienceTest- Daltix

# Technical Specification
* Physical
  * Dell Latitude 5490
  * Intel Core i5-8250U CPU @ 1.60Hz 1.80Hz
  * 8 GB RAM
* Software
  * Python 3.6.8
  * Sublime Text

# Libraries used
* Numpy: Numpy is free and open-source Python library used for scientific computing and technical computing.
* pandas: Pandas is a software library written for the Python programming language for data manipulation and analysis.
* Smart_open: for efficient streaming of very large files from local storage.
* Stop_words: Get list of common stop words in various languages.
* Sklearn: Machine learning library for the Python programming language.


# Approach
* In this test I first thought to match daltix_id if product id is same.While working on data i also realised that Every first word in name is mostly the brand name, it helps me to fill the missing values in Brand Column.

* After observing the y_true file i also realised that product id match is one factor but brand name should also be another factor while matching the daltix id.After applying both the condition 27213 are true and 3238 are incorrect matches.

* Then i approach to name column, apply Term Frequency - inverse document frequency & Cosine similarity to find the similarity between the names. Then it created a huge sparse matrix. To handle such huge matrix i have reduced the memory usage by changing the data type to float16 and convert it to a dense matrix.

* Create a new data set and add daltix_id corresponds to columns like brand, name & shop along with the cosine distance. Then i apply a condition with the approach that a particular product with same brand can be available in different shops.

* Then i evaluate the result with F1 Score by taking the multiple distances - between 0.3 to 0.9.


# Other Approaches which I had tried
Sparse Matrix is so huge that it crashed my system multiple time, also while converting sparse matrix to dense matrix it throughs memory Error. I tried to run it in batches but again it crashed my system.

# Results

