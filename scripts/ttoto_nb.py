
import nltk
import string

def extract_features(phrase):
    """
    Currently extracted features:
        - letter presence
        - biletter presence
        - length
    """
    
    punc=(",./<>;:@#'?&-{}[]()+=!$%^&*")
        
    cleantext = phrase.translate(string.maketrans("",""), punc)
    cleantext=cleantext.lower()
    cleantext=cleantext.replace("  "," ")
    
    
    
    letters = list(cleantext)
    length=len(cleantext)
    cleantext="^"+cleantext+"$"
    biletters =[cleantext[i:i+2] for i in range(len(cleantext)-1)]
    
    features = dict([(l,True) for l in letters+biletters])
    features['length']=length
    return features


cat1="pastas"
cat2="starWarsLocations"


cat1_handle = open('../data/'+cat1+'.txt')
cat2_handle = open('../data/'+cat2+'.txt')

training_data = []
test_data= []

cat1_list=cat1_handle.readlines()
cat2_list=cat2_handle.readlines()

cat1_list = [x.replace("\n","") for x in cat1_list]
cat2_list = [x.replace("\n","") for x in cat2_list]

# The \n is somehow becoming a part of the elements of the list. 
# I dunno if this is some linux-windows thing.
# This last statement is just to gt rid of that.


# I want both classes to be present in equal quantities
# So I'll limit my input list size to the size of the smaller of the two lists
maxlimit=min(len(cat1_list), len(cat2_list))

#create dataset and split into train test
import random
random.seed(45)
from sklearn import cross_validation
totaldata=[(word,cat1) for word in cat1_list[:maxlimit]] + [(word,cat2) for word in cat2_list[:maxlimit]]
traindata,testdata=cross_validation.train_test_split(totaldata,train_size=0.75,test_size=0.25,random_state=21)   

print maxlimit, len(traindata), len(testdata)


train_feature_set = [(extract_features(line), label) for (line, label) in traindata]
test_feature_set  = [(extract_features(line), label) for (line, label) in testdata]

classifier = nltk.NaiveBayesClassifier.train(train_feature_set)

#print classifier.most_informative_features(10)
print classifier.show_most_informative_features()



confusion ={(cat1,cat1):0, (cat1,cat2):0, (cat2,cat1):0, (cat2,cat2):0}
for t in testdata:

    pred=classifier.classify(extract_features(t[0]))
    actual=t[1]
    
    confusion[(pred,actual)]+=1
    
    # If you need the incorrectly predicted items, uncomment the next line
    if pred != actual:
        print t[0],actual, pred
    

print "--------"
print confusion


correct = sum([confusion[(x,y)] for (x,y) in confusion if x==y])
incorrect=sum([confusion[(x,y)] for (x,y) in confusion if x!=y])
accuracy=1.0*correct/(correct+incorrect)

print correct, incorrect,accuracy

print "--------"
print classifier.classify(extract_features("MyExample"))  #case insensitive



    
