# Market_Basket_Analysis

Association Rules and Lift Analysis

Proposal of Question

What associations are there in the dataset as they pertain to those customers who experienced churn?

Defined Goal

The goal of this project is to perform a Market Basket Analysis to answer the above research question. The Apriori Algorithm will assist in the accomplishment of this analysis.

Explanation of Market Basket

Market Basket Analysis (or MBA) is an econometric tool used to evaluate a dataset. The outcomes expected of any MBA is an understanding of certain associations which may or may not exist throughout a dataset’s transactions. A transaction means a single occurrence. Association should not be confused with correlation, as often happens, but should instead be thought of as rules for behavior amongst customer transactions. The classic example is a grocery store with individual customers and purchases. It is said that customers who buy diapers also buy beer, though unsupported. If we did in fact have a dataset with 10 customers, and 9 of those customers bought beer and diapers, the associative rule of: if beer then diapers, would be 0.9. How this applies to churn is via individual case orders. Each transaction which occurs in the dataset is a customer experience with the company and is represented by a case order. 
To perform the MBA, customers with positive churn will be grouped by case order and those variables proven to be highly correlated to churn. Then the Apriori algorithm will be used to find and narrow associated rules to only those rules in which there is a certain confidence, lift, and support. The expected outcome is a list of useful rules which may answer the research question and provide insight for strategic management initiatives in preventing customer churn.
Transaction Example
A transaction is a single instance which has occurred in a dataset representing a customer experience. The transaction will contain certain aspects such as: customer identifier, variables, features, and association throughout those variables. One transaction is shown below via screenshot with a customer case order, demographic information, and various variables (monthly charge and bandwidth).

Market Basket Assumption (MBA)

There are several assumptions made regarding Market Basket Analysis (MBA). One of those is the fact that there must be associations to be discerned within the data. With chaotic data, completely unassociated, an MBA would fail to provide insight into the dataset. Another is that the Apriori Algorithm only accepts 0 and 1 inputs. Thus, strings and non-transformed data become incompatible.

Transforming the Dataset

The analytical process of association rules can easily exceed computational power. Especially with such a large dataset. Since those customers who churned are the target of this analysis, the transformed dataset was reduced to only those customers who did experience churn. Furthermore, for the above-mentioned assumptions to be met, certain transformations were performed on the data. Those variables which are highly correlated to churn were split by quartile and given values 1 through 4. This allowed for a logical grouping of customers regarding monthly charge, bandwidth, and tenure. 
	
These transformations are suitable for Market Basket Analysis because they are easy to interpret. To understand the transformed results all that is required is to read the number assigned to the association rule and look at the quartile for that variable.

Code Execution

Below is the code necessary to perform the analysis with the screenshots demonstrating error-free execution.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from mlxtend.frequent_patterns import association_rules, apriori
from mlxtend.preprocessing import TransactionEncoder as ts
df = pd.read_csv('churn_data.csv')
display(df.head())
ddff = df
ddff.Bandwidth_GB_Year = ddff.Bandwidth_GB_Year.astype('float')
# Transform Bandwidth to Quartiles for perceivable groups within the market basket analysis
ddff['Bandwidth'] = 0
ddff.loc[(ddff['Bandwidth_GB_Year'] <=1236.47), 'Bandwidth'] = 1
ddff.loc[(ddff['Bandwidth_GB_Year'].between(1236.48, 3279.53)), 'Bandwidth'] = 2
ddff.loc[(ddff['Bandwidth_GB_Year'].between(3279.54, 5586.14)), 'Bandwidth'] = 3
ddff.loc[(ddff['Bandwidth_GB_Year'] >=5586.14), 'Bandwidth'] = 4

# Transform Tenure to Quartiles for perceivable groups within the market basket analysis
ddff['Tenure_T'] = 0
ddff.loc[(ddff['Tenure'] <=7.91), 'Tenure_T'] = 1
ddff.loc[(ddff['Tenure'].between(7.92, 35.43)), 'Tenure_T'] = 2
ddff.loc[(ddff['Tenure'].between(35.44, 61.48)), 'Tenure_T'] = 3
ddff.loc[(ddff['Tenure'] >=61.48), 'Tenure_T'] = 4

# Transform MonthlyCHarge to Quartiles for perceivable groups within the market basket analysis
ddff['Monthly'] = 0
ddff.loc[(ddff['MonthlyCharge'] <=139.98), 'Monthly'] = 1
ddff.loc[(ddff['MonthlyCharge'].between(139.99, 167.48)), 'Monthly'] = 2
ddff.loc[(ddff['MonthlyCharge'].between(167.48, 200.73)), 'Monthly'] = 3
ddff.loc[(ddff['MonthlyCharge'] >=200.74), 'Monthly'] = 4
basket = (ddff[ddff['Churn']=='Yes'].groupby(['CaseOrder', 'Tenure_T','Bandwidth','Monthly'])['CaseOrder'].sum().unstack().reset_index().fillna(0).set_index('CaseOrder'))
def one_hot_encoding(x):
    if x<=0:
        return 0
    if x>=1:
        return 1
baskets = basket.applymap(one_hot_encoding)
baskets
frequencies_of_itemsets =  apriori(baskets, min_support=(.07), use_colnames=True)
rules = association_rules(frequencies_of_itemsets, metric="lift", min_threshold=1)
rules
rules.confidence.describe()
rules[ (rules['lift']>1) & (rules['confidence']>=.7) & (rules['antecedent support']>.3)]




Association Rules Table

The values of support, lift, and confidence for the final reduced table are provided below:

Lift	X > 1
Support	X >= 0.07 
Confidence	X > 0.7

Top Three Rules

The association rules were reduced to these three through the criterion of their various scores throughout the summary of the algorithm. For example, rule 5 was listed as one of the top three because it has high antecedent/consequent support, support, confidence, and lift. This hints at the strength of the rule suggesting its applicability to the dataset.

Significance of Support, Lift, and Confidence Summary

Support

The support summary of the Apriori Algorithm is the frequency of occurrence of the rule. Using the grocery example, if 9 of 10 customers bought beer and diapers, then the support would be 0.9. The significance with high support lies in its actual representation of the dataset. With high lift, but low support, lift becomes suspect of an inaccurate representation of the data. Support is just another check on the algorithm’s inner workings regarding association rule accuracy. 

There are only three associative rules derived from the analysis with a minimum support of 0.07. Meaning that there must be at least a 7% frequency of the rules’ occurrence within the dataset. This eliminates unrealistic rules.

Lift

This is the summary of the likelihood of two items occurring together. If a rule has a lift score greater than 1, there is a higher likelihood of the antecedent and consequent occurring together. With a lift score of exactly 1, there is no association between the antecedent and consequent. The significance of this score is the further insight and reduction it provides to the association rules. 

There are only three associative rules derived from the analysis with a lift of X >1. Meaning, all rules which have been included in the final table (rule 5, 26, and 28) have a high probability of antecedents and consequents occurring together. 

Confidence

Confidence, a probability with the condition of antecedent and consequent occurring together, provides the probability that the association rule will occur by proportion. This allows for even further rule reduction. It is one of the main fine-tuning instruments to check the rules provided by the algorithm. 

The three rules found in the analysis have a final confidence interval of X >= 0.7. Meaning there is a confidence of at least 70% that any rule will occur. 

Practical Significance of Findings

The three rules provided by the analysis are:

This allows for the deduction that those customers with high tenure, bandwidth, and monthly charge (fourth quartile for each) are associated within positive customer churn. The recommendation is then intuitive that those customers with high tenure, bandwidth, and monthly premiums ought to be the focus of strategic management initiatives. The practical significance in these findings is the fact that there is high association amongst those customers found in the 4th quartile for each of the predictor variables listed. A rather intense, but insightful discovery regarding customer churn. 

Course of Action

The research question was, “What associations are there in the dataset as they pertain to those customers who experienced churn?” The answer, those customers who experience churn often associate to high monthly premiums, tenure, and bandwidth. How this assists the company’s need to reduce churn is it provides guidance to strategic management initiatives. Those customers identified by the derived association rules ought to be flagged and addressed to procure their continued business. Possible price decreases for these customers. Maybe cheaper contracts to loyal customers with high tenure is in order. This is entirely up to executive management as to the engagement of these groups. In summary, the research question posed has been answered, and association rules determined; thus, allowing for the possible reduction of customer churn.


