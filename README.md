# Model Description

---

## 1 Overview

This model combines a WCG (Woman-Child-Group) model (Figure 1) with XGBoost for survival prediction. The WCG model predicts survival for women and children (boys) at the family-group level based on group information, and this is added as a single feature. Combined with 6 additional features, XGBoost is used to predict passenger survival (Survived).

---

## 2 WCG Model: Concept and Implementation

The core hypotheses of the WCG model are:

- If all women and boys in a group survived, the boys in that group are likely to have survived.
- If all women and boys in a group died, the women in that group are likely to have died as well.

To implement these hypotheses, the following steps were used to engineer the features:

1. Extract the surname from the `Name` column to create a `Surname` column.
2. Combine `Surname`, `Pclass`, `Ticket` number (excluding the last character), `Fare`, and `Embarked` to create a `GroupId`.
3. Count group members; passengers in groups of 1 or less, or adult males (`Title=man`), are classified as `noGroup`.
4. Combine `Pclass`, `Ticket` number (excluding the last character), `Fare`, and `Embarked` to create a `TicketId`. This is used to include relatives who share a ticket but have different surnames in the same group.
5. Calculate the survival rate per group as `GroupSurvived`.
6. Build a dictionary mapping `GroupId` → `GroupSurvived` from the training data.
7. Use the dictionary to assign `GroupSurvived` values to the test data based on each passenger's `GroupId`.
8. For groups with an unknown survival rate, assign `GroupSurvived = 0` if `Pclass == 3` (lower survival class), and `GroupSurvived = 1` otherwise.
9. Construct the WCG variable as follows: initialize all rows to `WCG = 0` (predicted not to survive). Then set all women (`Title=woman`) to `WCG = 1` (predicted to survive) as the default. Next, override: women in groups where `GroupSurvived = 0` (all group members died) are set to `WCG = 0`; boys (`Title=boy`) in groups where `GroupSurvived = 1` (all group members survived) are set to `WCG = 1`. The `GroupSurvived` variable itself was not included in the final model, as adding it did not improve accuracy.

---

## 3 Features Used

The following features were engineered and fed into the model:

| Feature | Description |
|---|---|
| Fare | Raw fare value |
| TicketFreq | Number of passengers sharing the same ticket number |
| WCG | WCG model prediction (1 = predicted to survive, 0 = predicted to die) |
| FamilySize | SibSp + Parch + 1 |
| Sex_female | Dummy variable indicating whether the passenger is female |
| Pclass_1–3 | One-hot encoded Pclass variables |

---

## 4 Model and Parameter Settings

XGBoost was selected as the final model. The hyperparameters used are:

- `max_depth=3`
- `learning_rate=0.1`
- `n_estimators=80`

Ensemble methods combining other classifiers (Support Vector Machine, Multi-Layer Perceptron, etc.) were also tested, but they did not improve accuracy, so XGBoost alone was adopted.

---

## 5 Key Design Decisions

The original goal was to implement the hypothesis: *"If a group boarded together and all of them died, they would likely die in the test data too."* However, it was unclear how to implement this in code. While researching Kaggle notebooks, a similar approach was found and used as a reference ([Chris Deotte's notebook](https://www.kaggle.com/code/cdeotte/titanic-wcg-xgboost-0-84688)).

The reference notebook applied predictions only to male passengers, but this did not improve accuracy due to class imbalance. Therefore, in this model, the WCG-based prediction was added as a feature covering both women and boys. A variable indicating women who boarded alone was also tested but was excluded from the final model as it did not improve accuracy.

---

## Figure 1: WCG Model Decision Flow

```
Is the passenger an adult male?
│
├─ Yes (adult male)
│   └─ WCG = 0 (predicted to die)
│
└─ No (woman or boy)
    │
    ├─ Woman
    │   ├─ Default: WCG = 1 (predicted to survive)
    │   └─ All women and boys in the group died (GroupSurvived = 0)
    │       └─ WCG = 0 (predicted to die)
    │
    └─ Boy
        ├─ Default: WCG = 0 (predicted to die)
        └─ All women and boys in the group survived (GroupSurvived = 1)
            └─ WCG = 1 (predicted to survive)
```

> **Reference notebook:** https://www.kaggle.com/code/cdeotte/titanic-wcg-xgboost-0-84688
