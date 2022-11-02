Question 1:
For question 1, the array size's I ran were 100,000 and 1,000,000.  
The array's had to be this large so time.time() would return a value that is not 0.00. When we ran our pytest script for the
non vectorized and vectorized test cases with array size of 100,000 and 1,000,000 we got the following result:

Non vectorized for-loop implementation:
Array Size:  Time: 
100,000      0.03100609
1,000,000    0.2965958

vectorized non for-loop implementation:
Array Size:  Time: 
100,000      0.0010001
1,000,000    0.0030097

We can see from the above results that the vectorized non for-loop implementation is significantly faster (30x and 97X) at computing the euclidean 
distance for array size 100,000 and 1,000,0000. The reason this implementation is significantly faster is due to the fact that vectorized 
code can implemented using Single Instruction, multiple data (SIMD) code. What SIMD code can do is it can allow for more bits of data to be 
stored in cache space and thus more data can be ran per instruction cycle by the CPU (compared to a for loop implementation). In addition, since vectorized 
code uses multiple functions it can batch processes (i.e sum all numbers in an array instead of subtracting, squaring then adding this number to the running total). 
This can optimize CPU usage by reducing CPU warm up time. Lastly, vectorized code can be parallelized and run on the GPU, more efficiently than a for loop.

Question 2:
We can see from the graphs that are attached in this zip file that both KNN-algorithms preformed decently well. When the Cosine function was not used, the training and validation score both ended up in the low 70's range which indicates a solid prediction algorithm. When we used our cosine function to predict whether a headline was a real or faked the KNN accuracy score jumped up to the low 80's for training and high 70's for validation. The reason for this increase in accuracy was due to cosine's ability to better map high dimensional items. 

Cosine can better detect items in high dimension since the cosine function relies on the angle between two vectors. This is important since angles are a finite length while distance can hypothetically being infinite. For example, In a high dimension space one document can appear closer to another due to the fact they share more words. But a larger document/headline will inherently share more words compared to a smaller headline. Because of this, we don't want to focus on quantity of words since this will be bias towards larger documents. Therefore, we will want to focus on how common certain words are per document. This is better predicted by cosine instead of Euclidean distance since how far something is in one dimension doesn’t affect the cosine. 

Both functions tend to do better when there are more "Neighbours" to work with. The reason for this is there more data to work with which minimizes randomness. Sometimes when documents are projected into vectors, they may appear to be closer to another class by mistake. This randomness is minimized where there are more data points to compare the "closet/max democratic function" to. In addition, the model tends to do better when the "Neighbours classifier" is odd, since there must be a definitive winner, whereas an even number can end up in 50/50 tie. 

Question 3:

I implemented the Naïve Bayes function using the Lapace smoothing algorithm. In addition, I also used logs to calculate the sum of the class score instead of using the product of integer values since this would minimize numerical overflow as python can only handle numbers so small. 
I gave the user the option of using or removing stop words. I did not go through the stop word list myself and remove any words I thought should be included in the vocabulary for the naïve bayes function. Furthermore, I did not do anything to counter act the fact that words in headlines are not truly independent. For example, Don Trump will be two distinct words even though they should be combined into one. By accounting for this the model could have performed better.  
High level, the model works by taking the file path to two training documents. From here we parse through each document line by line and count each word in each class while taking the union of words for both classes. From here we calculate our log priors and our loglikehood by appending tuples (that are based on the word and its associated class) into a dictionary. In this training portion of our code, we make our assumption about our model. For example, we used logs, and Lapace smoothing to make the numbers more readable, and we choose to use stopwords or not to classify our classes.   
 Once we have this our training is complete. Next, we can predict and evaluate our model by. At this point we will call our evaluate method which will call predict and score for each document/headline we pass it. The predict method will tell us which class it thinks the document is. From here, we total the true positives/false positives by comparing our predictive class with the documents/headlines from the true testing data, and we do the same for true negatives for fake testing set. 
The results of my model are below:
Performance on class <RELEVANT>, keeping stopwords
TP: 255
FP: 40
TN: 155
FN: 40
	Precision: 0.864406779661017	 Recall: 0.864406779661017	 F1: 0.864406779661017

Performance on class <RELEVANT>, removing stopwords
TP: 261
FP: 34
TN: 136
FN: 59
	Precision: 0.8847457627118644	 Recall: 0.815625	 F1: 0.8487804878048781

Top features for class <FAKE.TXT>
	breaking	19.857009425150714
	3	17.650675044578406
	soros	15.444340664006104
	black	14.892757068863025
	watch	14.341173473719957
	7	13.238006283433807
	woman	13.238006283433807
	secret	11.031671902861502
	duke	11.031671902861502
	steal	11.031671902861502

Top features for class <REAL.TXT>
	ban	52.57589285714284
	korea	48.04348830049259
	travel	38.07219827586206
	turnbull	35.352755541871886
	trumps	33.99303417487686
	australia	25.381465517241363
	climate	21.755541871921174
	north	17.525297619047624
	paris	15.410175492610824
	refugee	14.50369458128078

We can see our model did a solid job when predicting whether a document was real. It struggled a little bit when trying to accurately predict if the document was fake. I believe the reason for this is we didn’t isolate words like donald trump vs President Trump (trump will come up in both real and fake articles). In this example, if we score for both of these words together, we could better predict whether a document is real or not since Real articles are more likely to be formal compared to fake articles. Also for predicting fake headlines, checking for misspelled words could improve accuracy. 

