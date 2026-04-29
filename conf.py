import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Define the class labels in order
classes = ['access_control', 'integer_overflow', 'logic_error', 
           'other', 'reentrancy', 'safe']

# 2. Replicate the matrix data from the image
# Rows = Actual, Columns = Predicted
data = [
    
    [19, 0, 1, 4, 0, 4], # access_control
    [0, 20, 0, 0, 1, 0], # integer_overflow
    [2, 0, 13, 0, 1, 0], # logic_error
    [0, 0, 0, 18, 0, 2], # other
    [1, 1, 0, 0, 18, 0], # reentrancy
    [0, 2, 0, 2, 0, 16]  # safe
]  # safe


# 3. Create a DataFrame for easier labeling
df_cm = pd.DataFrame(data, index=classes, columns=classes)

# 4. Plotting
plt.figure(figsize=(8, 6))
sns.set_context("notebook", font_scale=1.1)

# cmap="Blues" matches your image color scale
# annot=True puts the numbers in the boxes
ax = sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=True)

# 5. Fine-tuning labels and titles
plt.title('Confusion Matrix — Multi-class', fontweight='bold', fontsize=16, pad=20)
plt.xlabel('Predicted', fontsize=12, labelpad=10)
plt.ylabel('Actual', fontsize=12, labelpad=10)

# Rotate labels for better readability as seen in your image
plt.xticks(rotation=30, ha='right')
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()