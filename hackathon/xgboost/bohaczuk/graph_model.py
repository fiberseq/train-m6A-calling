# Takes .json file from XGBoost model and generates a bar graph of feature weight.

import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Load XGBoost model
bst = xgb.Booster()
bst.load_model("xgboost.otherhalf.hifi.100rds.json")

# Get feature score and pull out weight values
feature_scores = bst.get_score(importance_type='weight')
values = list(feature_scores.values())

# Create a list of keys and pull out only the ones with weight >0 in the model
feature_scores_stripped=[ int(f.strip("f")) for f in feature_scores]

keys = ['A'+str(n) for n in range(1,16)] + ['C'+str(n) for n in range(1,16)]+ ['G'+str(n) for n in range(1,16)] \
	+ ['T'+str(n) for n in range(1,16)] + ['IP'+str(n) for n in range(1,16)] + ['PW'+str(n) for n in range(1,16)]
# + ['Offset'+str(n) for n in range(1,16)]

keys_used=[]
for x in feature_scores_stripped:
	addkey = keys[x]
	keys_used.append(addkey)

# Generate graph
plt.figure(figsize = (20,10))
plt.xticks(rotation=45)
ax= sns.barplot(x=keys_used, y=values)
ax.set_ylabel("weights")
plt.savefig("xgb.testgraph.pdf", format='pdf', bbox_inches="tight")
