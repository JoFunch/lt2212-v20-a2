Pt1. 

the preprocessing done in part 1 includes tokenisation on whitespace, lowering capital, and removing numbers and symbols. In the code, you'll also find two additional preprocessing strings, but those I took away as they completely dropped the overall classification rate. 

Of course, these could have been included, which would have improved the time overall, but the results where slightly better without. 

pt 2

in the original script I used SVD where I in the bonus part used PCA. 

pt 3 

I used KneightborsClassifier and Naieve Bayes.


Following is the Accuracy, Precision, recall, and F-Measure of respectively KNeightborsClassifier, NaiveBayes, and DecisionTreeClassifier. 



Part 4

KNeighborsClassifier: 

Accuracy:  0.36074270557029176
Precision:  0.42539017448796657
recall:  0.35858746911955525
F-Measure:  0.36878891330406305


   macro avg   precision    0.43     
weighted avg   precision    0.42


GaussianNB:

Part 4

Accuracy:  0.7840848806366048
Precision:  0.7810080602806917
recall:  0.7812991445206692
F-Measure:  0.7794538461036141

   micro avg       0.78      
   macro avg       0.79      
weighted avg       0.79

______



Part 2 : Dimension Reduction
KNeighborsClassifier: 

1000 dims

Accuracy:  0.35119363395225467
Precision:  0.3944648770759483
recall:  0.34902136136974315
F-Measure:  0.35631061609521597

weighted avg    0.39
macro avg       0.39
micro avg       0.35


500 dims

Accuracy:  0.32652519893899207
Precision:  0.3638358860067592
recall:  0.3224298594295505
F-Measure:  0.32773164682021044

micro avg       0.33     
macro avg       0.36    
weighted avg    0.36

100

Accuracy:  0.2596816976127321
Precision:  0.28666938401614084
recall:  0.2576973034792129
F-Measure:  0.25805858738024634

micro avg       0.26      
macro avg       0.29      
weighted avg       0.28      

50
Accuracy:  0.2116710875331565
Precision:  0.23158588668140742
recall:  0.2124250243834625
F-Measure:  0.20998849832866964

   micro avg       0.21      
   macro avg       0.23      
weighted avg       0.23

25

Accuracy:  0.18859416445623342
Precision:  0.20685958564412302
recall:  0.1850404606792142
F-Measure:  0.18518210753269518

   micro avg       0.19     
   macro avg       0.21    
weighted avg       0.21

10

Accuracy:  0.13607427055702917
Precision:  0.15513412482454467
recall:  0.13742713357849895
F-Measure:  0.13605680942491366

   micro avg       0.14     
   macro avg       0.16   	
weighted avg       0.15 



_________


GaussianNB:

Part 2 : -- Dims

1000

Accuracy:  0.1506631299734748
Precision:  0.2984541330724396
recall:  0.1458518873130346
F-Measure:  0.13439574213279187

   micro avg       0.15      
   macro avg       0.30      
weighted avg       0.30


500

Accuracy:  0.15172413793103448
Precision:  0.25378986350338667
recall:  0.14640659093543346
F-Measure:  0.13325309253376155

   micro avg       0.15     
   macro avg       0.25   
weighted avg       0.26

100

Accuracy:  0.14668435013262598
Precision:  0.18623494498960608
recall:  0.1404750284848092
F-Measure:  0.1090592456641849

   micro avg       0.15      
   macro avg       0.19      
weighted avg       0.19

50

Accuracy:  0.11750663129973475
Precision:  0.17214348574387145
recall:  0.11911240608034306
F-Measure:  0.09222440820549754

   micro avg       0.12      
   macro avg       0.17      
weighted avg       0.18

25

Accuracy:  0.10954907161803713
Precision:  0.13318518561650536
recall:  0.10708751907207956
F-Measure:  0.08089462669519902

   micro avg       0.11      
   macro avg       0.13     
weighted avg       0.13



10

Accuracy:  0.10344827586206896
Precision:  0.10807676027148769
recall:  0.09842247189346343
F-Measure:  0.068969706248238

   micro avg       0.10     
   macro avg       0.11     
weighted avg       0.11




_____

Bonus: 
PCA instead of SVD 

KNeighborsClassifier
 no reduction

Accuracy:  0.356763925729443
Precision:  0.4243483125674448
recall:  0.3545013786260134
F-Measure:  0.364863111629578

micro avg          0.36      
macro avg          0.42      
weighted avg       0.42

1000

Accuracy:  0.3506631299734748
Precision:  0.3969953034579255
recall:  0.34732195182452763
F-Measure:  0.3528234199823011

   micro avg       0.35     
   macro avg       0.40      
weighted avg       0.40

500

Accuracy:  0.32148541114058354
Precision:  0.3563390102015113
recall:  0.3190944181617683
F-Measure:  0.3235793566570823

   micro avg       0.32      
   macro avg       0.36      
weighted avg       0.36

100

Accuracy:  0.253315649867374
Precision:  0.2837534053566474
recall:  0.25123193189365356
F-Measure:  0.25284461794548757

   micro avg       0.25      
   macro avg       0.28      
weighted avg       0.29

50

Accuracy:  0.21962864721485412
Precision:  0.24182825067551902
recall:  0.2176554602343365
F-Measure:  0.21718050560493357


   micro avg       0.22      
   macro avg       0.24      
weighted avg       0.25

25

Accuracy:  0.19018567639257294
Precision:  0.2033133454861892
recall:  0.18603215694039596
F-Measure:  0.18461765317592121


   micro avg       0.19      
   macro avg       0.20      
weighted avg       0.21

10


Accuracy:  0.14058355437665782
Precision:  0.15154943383529945
recall:  0.1384169577317447
F-Measure:  0.13639201739580056


   micro avg       0.14      
    macro avg      0.15     
weighted avg       0.15


GaussianNB

500

Accuracy:  0.16127320954907162
Precision:  0.2741685413011943
recall:  0.15230333640854354
F-Measure:  0.141600438288851

   micro avg       0.16      
   macro avg       0.27      
weighted avg       0.27

100
Accuracy:  0.1464190981432361
Precision:  0.20664946844217041
recall:  0.14451389370590378
F-Measure:  0.11925305839178409

   micro avg       0.15     
   macro avg       0.21     
weighted avg       0.21

50

Accuracy:  0.12175066312997347
Precision:  0.1562988181644729
recall:  0.11964953069489952
F-Measure:  0.09485836614068628

   micro avg       0.12      
   macro avg       0.16     
weighted avg       0.16

25

Accuracy:  0.10557029177718832
Precision:  0.1360738286438728
recall:  0.10688254301064617
F-Measure:  0.08131105078736581

   micro avg       0.11      
   macro avg       0.14     
weighted avg       0.14

10

Accuracy:  0.09363395225464191
Precision:  0.11434426880204282
recall:  0.09319618556620782
F-Measure:  0.06685692286022539

   micro avg       0.09      
   macro avg       0.11      
weighted avg       0.12

