# DataScienceTest- Daltix

#Libraries used
1. Numpy: Numpy is free and open-source Python library used for scientific computing and technical computing.
2. pandas: Pandas is a software library written for the Python programming language for data manipulation and analysis.
3. Smart_open: 
4. Stop_words:
5. Sklearn: Machine learning library for the Python programming language.


#Approach
1. In this test I first thought to match daltix_id if product id is same.While working on data i also realised that Every first word in name is mostly the brand name, it helps me to fill the missing values in Brand Column.

2. After observing the y_true file i also realised that product id match is one factor but brand name should also be another factor while matching the daltix id.After applying both the condition 27213 are true and 3238 are incorrect matches.

3. Then i approach to name column, apply Term Frequency - inverse document frequency & Cosine similarity to find the similarity between the names. Then it created a huge sparse matrix. To handle such huge matrix i have reduced the memory usage by changing the data type to float16 and convert it to a dense matrix.

4. Create a new data set and add daltix_id corresponds to columns like brand, name & shop along with the cosine distance. Then i apply a condition with the approach that a particular product with same brand can be available in different shops.

5. Then i evaluate the result with F1 Score by taking the multiple distances - 0.3 to 0.9.
