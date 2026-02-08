# Machine Learning Workshops 1 & 2 - VIVA Preparation Guide

## Table of Contents
1. [Workshop 1: NumPy Fundamentals](#workshop-1-numpy-fundamentals)
2. [Workshop 2: Pandas Fundamentals](#workshop-2-pandas-fundamentals)
3. [Quick Reference](#quick-reference)

---

## Workshop 1: NumPy Fundamentals

### 1. **Array Basics**

#### `np.array()`
- **Purpose**: Creates a NumPy array from Python list/tuple
- **Syntax**: `np.array(data)`
- **Example**:
  ```python
  arr = np.array([1, 2, 3])  # 1D array
  arr_2d = np.array([[1, 2], [3, 4]])  # 2D array
  ```

#### `.ndim` (Attribute)
- **Purpose**: Returns the number of dimensions
- **Syntax**: `array.ndim`
- **Example**: `arr.ndim` → Returns `2` for 2D array

#### `.shape` (Attribute)
- **Purpose**: Returns dimensions as tuple (rows, columns)
- **Syntax**: `array.shape`
- **Example**: `arr.shape` → Returns `(2, 3)` for 2 rows, 3 columns

#### `.dtype` (Attribute)
- **Purpose**: Returns data type of array elements
- **Syntax**: `array.dtype`
- **Example**: `arr.dtype` → Returns `dtype('int64')`

#### `.reshape()`
- **Purpose**: Changes array shape without modifying data
- **Syntax**: `array.reshape(new_shape)`
- **Example**:
  ```python
  arr = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
  reshaped = arr.reshape(3, 2)  # (3, 2)
  ```

---

### 2. **Array Creation Functions**

#### `np.arange()`
- **Purpose**: Creates array with evenly spaced values (like Python's `range()`)
- **Syntax**: `np.arange(start, stop, step)`
- **Examples**:
  ```python
  np.arange(10)           # [0, 1, 2, ..., 9]
  np.arange(1, 10)        # [1, 2, 3, ..., 9]
  np.arange(0, 10, 2)     # [0, 2, 4, 6, 8]
  np.arange(0.5, 10.4, 0.8)  # Works with floats
  ```

#### `np.linspace()`
- **Purpose**: Creates array with specified number of evenly spaced values
- **Syntax**: `np.linspace(start, stop, num=50, endpoint=True)`
- **Examples**:
  ```python
  np.linspace(1, 10)          # 50 values between 1 and 10
  np.linspace(1, 10, 7)       # 7 values between 1 and 10
  np.linspace(1, 10, 7, endpoint=False)  # Excludes 10
  ```
- **Key Difference from arange**: Specify COUNT of elements, not STEP size

#### `np.ones()`
- **Purpose**: Creates array filled with ones
- **Syntax**: `np.ones(shape)`
- **Example**:
  ```python
  np.ones((2, 3))  # 2x3 array of ones
  np.ones(arr.shape)  # Same shape as existing array
  ```

#### `np.zeros()`
- **Purpose**: Creates array filled with zeros
- **Syntax**: `np.zeros(shape)`
- **Example**: `np.zeros((3, 2))  # 3x2 array of zeros`

#### `np.empty()`
- **Purpose**: Creates uninitialized array (faster but contains garbage values)
- **Syntax**: `np.empty(shape)`
- **Example**: `np.empty((2, 2))`

#### `np.eye()`
- **Purpose**: Creates identity matrix (diagonal of 1s)
- **Syntax**: `np.eye(n)`
- **Example**:
  ```python
  np.eye(3)  # 3x3 identity matrix
  # [[1, 0, 0],
  #  [0, 1, 0],
  #  [0, 0, 1]]
  ```

#### `np.full()`
- **Purpose**: Creates array filled with specified value
- **Syntax**: `np.full(shape, fill_value)`
- **Example**: `np.full((3, 3), 7)  # 3x3 array filled with 7`

#### `np.zeros_like()` / `np.ones_like()`
- **Purpose**: Creates array of zeros/ones with same shape and type as given array
- **Syntax**: `np.zeros_like(array)`
- **Example**:
  ```python
  arr = np.array([[1, 2, 3], [4, 5, 6]])
  np.zeros_like(arr)  # Same shape (2, 3), filled with 0
  ```

---

### 3. **Array Manipulation**

#### `np.copy()`
- **Purpose**: Creates independent copy of array
- **Syntax**: `np.copy(array)`
- **Example**: `copied = np.copy(arr)`

#### `np.concatenate()`
- **Purpose**: Joins arrays along existing axis
- **Syntax**: `np.concatenate((arr1, arr2), axis=0)`
- **Examples**:
  ```python
  arr1 = np.array([[1, 2], [3, 4]])
  arr2 = np.array([[5, 6], [7, 8]])
  
  np.concatenate((arr1, arr2), axis=0)  # Vertical (rows)
  # [[1, 2], [3, 4], [5, 6], [7, 8]]
  
  np.concatenate((arr1, arr2), axis=1)  # Horizontal (cols)
  # [[1, 2, 5, 6], [3, 4, 7, 8]]
  ```

#### `np.vstack()`
- **Purpose**: Stacks arrays vertically (row-wise)
- **Syntax**: `np.vstack((arr1, arr2))`
- **Example**: Same as `concatenate(..., axis=0)`

#### `np.column_stack()`
- **Purpose**: Stacks 1D arrays as columns into 2D array
- **Syntax**: `np.column_stack((arr1, arr2))`
- **Example**:
  ```python
  v = np.array([9, 10])
  w = np.array([11, 12])
  np.column_stack((v, w))  # [[9, 11], [10, 12]]
  ```

---

### 4. **Indexing and Slicing**

#### Basic Indexing
- **Syntax**: `array[index]` or `array[row, col]`
- **Examples**:
  ```python
  arr1d[2]           # Element at index 2
  arr2d[1, 2]        # Row 1, Column 2
  arr3d[1, 0, 1]     # 3D indexing
  ```

#### Slicing
- **Syntax**: `array[start:stop:step]`
- **Examples**:
  ```python
  arr1d[1:4]         # Elements from index 1 to 3
  arr1d[1:5:2]       # Elements at indices 1, 3 (step=2)
  arr2d[1:3, 0:2]    # Rows 1-2, Columns 0-1
  arr2d[::2, 1::2]   # Every 2nd row, every 2nd col starting at 1
  arr[::-1]          # Reverse array
  ```

#### Advanced Indexing
- **Border assignment**: `arr[1:-1, 1:-1] = 0`  # Set interior to 0
- **Checkerboard pattern**: 
  ```python
  arr[::2, 1::2] = 1  # Odd rows, even columns
  arr[1::2, ::2] = 1  # Even rows, odd columns
  ```

---

### 5. **Array Operations**

#### Arithmetic Operations (Element-wise)
- **Addition**: `arr1 + arr2`
- **Subtraction**: `arr1 - arr2`
- **Multiplication**: `arr1 * 3`
- **Power**: `arr ** 2`
- **Examples**:
  ```python
  x = np.array([[1, 2], [3, 5]])
  y = np.array([[5, 6], [7, 8]])
  
  x + y    # [[6, 8], [10, 13]]
  x - y    # [[-4, -4], [-4, -3]]
  x * 3    # [[3, 6], [9, 15]]
  x ** 2   # [[1, 4], [9, 25]]
  ```

#### `np.dot()` (Dot Product)
- **Purpose**: Computes dot product / matrix multiplication
- **Syntax**: `np.dot(arr1, arr2)` or `arr1 @ arr2`
- **Examples**:
  ```python
  v = np.array([9, 10])
  w = np.array([11, 12])
  np.dot(v, w)  # 9*11 + 10*12 = 219
  
  x = np.array([[1, 2], [3, 5]])
  y = np.array([[5, 6], [7, 8]])
  np.dot(x, y)  # Matrix multiplication
  ```

---

### 6. **Linear Algebra (`np.linalg` module)**

#### `np.linalg.inv()`
- **Purpose**: Computes matrix inverse
- **Syntax**: `np.linalg.inv(matrix)`
- **Example**:
  ```python
  A = np.array([[3, 4], [7, 8]])
  A_inv = np.linalg.inv(A)
  ```

#### `np.linalg.solve()`
- **Purpose**: Solves system of linear equations (AX = B)
- **Syntax**: `np.linalg.solve(A, B)`
- **Example**:
  ```python
  A = np.array([[2, -3, 1], [1, -1, 2], [3, 1, -1]])
  B = np.array([-1, -3, 9])
  X = np.linalg.solve(A, B)  # More efficient than using inverse
  ```

#### `.T` (Transpose)
- **Purpose**: Returns transposed array
- **Syntax**: `array.T`
- **Example**:
  ```python
  arr = np.array([[1, 2], [3, 4]])
  arr.T  # [[1, 3], [2, 4]]
  ```

#### `np.linalg.det()`
- **Purpose**: Computes determinant
- **Syntax**: `np.linalg.det(matrix)`

#### `np.linalg.matrix_rank()`
- **Purpose**: Returns rank of matrix
- **Syntax**: `np.linalg.matrix_rank(matrix)`

#### `np.linalg.eig()`
- **Purpose**: Computes eigenvalues and eigenvectors
- **Syntax**: `eigenvalues, eigenvectors = np.linalg.eig(matrix)`

#### `np.linalg.norm()`
- **Purpose**: Computes matrix/vector norm
- **Syntax**: `np.linalg.norm(array)`

---

### 7. **Statistical Functions**

#### `.mean()`, `.median()`, `.min()`, `.max()`
- **Purpose**: Statistical aggregations
- **Syntax**: `array.mean()` or `np.mean(array)`
- **Examples**:
  ```python
  arr = np.random.random(30)
  arr.mean()    # Average value
  arr.min()     # Minimum value
  arr.max()     # Maximum value
  ```

---

### 8. **Random Number Generation**

#### `np.random.random()`
- **Purpose**: Generates random floats between 0 and 1
- **Syntax**: `np.random.random(size)`
- **Example**:
  ```python
  np.random.random(30)      # 1D array of 30 random values
  np.random.random((10, 10))  # 10x10 array
  ```

#### `np.random.seed()`
- **Purpose**: Sets random seed for reproducibility
- **Syntax**: `np.random.seed(number)`
- **Example**: `np.random.seed(42)`

---

### 9. **Utility Functions**

#### `np.allclose()`
- **Purpose**: Checks if two arrays are element-wise equal (within tolerance)
- **Syntax**: `np.allclose(arr1, arr2)`
- **Example**: `np.allclose(A @ A_inv, np.eye(2))  # True`

#### `np.array_equal()`
- **Purpose**: Checks if arrays are exactly equal
- **Syntax**: `np.array_equal(arr1, arr2)`

#### `np.c_[]` (Column concatenation)
- **Purpose**: Concatenates along second axis
- **Syntax**: `np.c_[arr1, arr2]`

---

## Workshop 2: Pandas Fundamentals

### 1. **Data Structures**

#### `pd.Series()`
- **Purpose**: 1D labeled array
- **Syntax**: `pd.Series(data, index=None)`
- **Examples**:
  ```python
  series = pd.Series([10, 20, 30, 40])
  series_indexed = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
  ```

#### `pd.DataFrame()`
- **Purpose**: 2D labeled data structure (table)
- **Syntax**: `pd.DataFrame(data, index=None, columns=None)`
- **Examples**:
  ```python
  # Style 1: Dictionary of lists
  df = pd.DataFrame({'Name': ['Alice', 'Bob'], 'Age': [25, 30]})
  
  # Style 2: With custom index
  df = pd.DataFrame(
      {'Bob': ['I liked it.', 'It was awful.'], 
       'Sue': ['Pretty good.', 'Bland.']},
      index=['Product A', 'Product B']
  )
  ```

---

### 2. **Index Management**

#### `pd.date_range()`
- **Purpose**: Creates date range
- **Syntax**: `pd.date_range(start, periods)`
- **Example**:
  ```python
  dates = pd.date_range('2023-01-01', periods=3)
  series = pd.Series([10, 20, 30], index=dates)
  ```

#### `.index` (Attribute)
- **Purpose**: Accesses/sets index
- **Syntax**: `df.index`
- **Example**: `series.index = ['x', 'y', 'z']`

#### `.reset_index()`
- **Purpose**: Resets index to default integer index
- **Syntax**: `df.reset_index(inplace=True)`
- **Example**:
  ```python
  df = pd.DataFrame({'A': [1, 2]}, index=['row1', 'row2'])
  df.reset_index(inplace=True)
  ```

---

### 3. **Reading and Writing Data**

#### `pd.read_csv()`
- **Purpose**: Reads CSV file into DataFrame
- **Syntax**: `pd.read_csv(filepath)`
- **Example**:
  ```python
  df = pd.read_csv('bank.csv')
  df = pd.read_csv('/path/to/file.csv')
  ```

#### `.to_csv()`
- **Purpose**: Writes DataFrame to CSV file
- **Syntax**: `df.to_csv(filepath, index=False)`
- **Example**:
  ```python
  df.to_csv('output.csv', index=False)
  df.to_csv('bank_numeric_data.csv', index=False)
  ```

---

### 4. **Data Inspection**

#### `.head()` / `.tail()`
- **Purpose**: Views first/last n rows
- **Syntax**: `df.head(n)` / `df.tail(n)`
- **Default**: n=5
- **Example**:
  ```python
  df.head()     # First 5 rows
  df.head(2)    # First 2 rows
  df.tail(1)    # Last row
  ```

#### `.info()`
- **Purpose**: Displays DataFrame structure (columns, dtypes, null counts)
- **Syntax**: `df.info()`
- **Output**: Shows column names, non-null counts, data types, memory usage

#### `.describe()`
- **Purpose**: Generates summary statistics
- **Syntax**: `df.describe()`
- **Output**: count, mean, std, min, 25%, 50%, 75%, max (for numerical columns)

#### `.shape` (Attribute)
- **Purpose**: Returns dimensions (rows, columns)
- **Syntax**: `df.shape`
- **Example**: `df.shape` → `(891, 12)` means 891 rows, 12 columns

#### `.columns` (Attribute)
- **Purpose**: Returns column names
- **Syntax**: `df.columns`
- **Example**: `df.columns.tolist()`

#### `.dtypes` (Attribute)
- **Purpose**: Returns data types of columns
- **Syntax**: `df.dtypes`

---

### 5. **Data Selection**

#### Column Selection
- **Single column**: `df['column_name']` → Returns Series
- **Multiple columns**: `df[['col1', 'col2']]` → Returns DataFrame
- **Example**:
  ```python
  df['Age']                    # Series
  df[['Name', 'Salary']]       # DataFrame
  ```

#### `.iloc[]` (Integer-based indexing)
- **Purpose**: Selects by integer position
- **Syntax**: `df.iloc[row_index]` or `df.iloc[row, col]`
- **Examples**:
  ```python
  df.iloc[0]        # First row
  df.iloc[0:3]      # First 3 rows
  df.iloc[:, 0:2]   # All rows, first 2 columns
  ```

#### `.loc[]` (Label-based indexing)
- **Purpose**: Selects by label/condition
- **Syntax**: `df.loc[condition]`
- **Examples**:
  ```python
  df.loc[df['Age'] > 30]              # Conditional filtering
  df.loc[0:2, ['Name', 'Age']]        # By labels
  ```

---

### 6. **Data Filtering**

#### Boolean Indexing
- **Syntax**: `df[condition]`
- **Examples**:
  ```python
  df[df['Age'] > 28]                  # Age greater than 28
  df[df['Pclass'] == 1]               # First class only
  df[(df['Age'] > 30) & (df['Sex'] == 'male')]  # Multiple conditions
  ```

---

### 7. **Data Modification**

#### `.drop()`
- **Purpose**: Removes rows or columns
- **Syntax**: `df.drop(columns=['col']) or df.drop(index=idx)`
- **Examples**:
  ```python
  df.drop(columns=['Salary'])         # Drop column
  df.drop(index=1)                    # Drop row at index 1
  ```

#### Adding Columns
- **Syntax**: `df['new_col'] = values`
- **Example**:
  ```python
  df['Bonus'] = df['Salary'] * 0.1
  ```

#### `.rename()`
- **Purpose**: Renames columns/index
- **Syntax**: `df.rename(columns={'old': 'new'})`
- **Example**:
  ```python
  df.rename(columns={'Name': 'FullName', 'Age': 'Years'})
  ```

---

### 8. **Handling Missing Data**

#### `.isnull()` / `.isna()`
- **Purpose**: Detects missing values
- **Syntax**: `df.isnull()`
- **Example**:
  ```python
  df.isnull().sum()    # Count nulls per column
  missing = df.isnull().sum()
  print(missing[missing > 0])  # Only columns with nulls
  ```

#### `.fillna()`
- **Purpose**: Fills missing values
- **Syntax**: `df.fillna(value)`
- **Examples**:
  ```python
  df.fillna(0)                    # Fill with 0
  df.fillna(df.mean())            # Fill with mean
  df.fillna(df.median())          # Fill with median
  df.fillna(df.mode()[0])         # Fill with mode
  df['col'].fillna(value, inplace=True)  # In-place
  ```

#### `.ffill()` (Forward fill)
- **Purpose**: Propagates last valid value forward
- **Syntax**: `df.ffill()`
- **Example**: `df_ffill = df.ffill()`

#### `.dropna()`
- **Purpose**: Removes rows/columns with missing values
- **Syntax**: `df.dropna(subset=['col'])`
- **Example**:
  ```python
  df.dropna()                 # Drop any row with null
  df.dropna(subset=['Age'])   # Drop rows where Age is null
  ```

---

### 9. **Data Cleaning**

#### `.str.strip()`
- **Purpose**: Removes whitespace from strings
- **Syntax**: `df['col'].str.strip()`
- **Example**:
  ```python
  df['Name'] = df['Name'].str.strip()
  ```

#### `.astype()`
- **Purpose**: Converts data type
- **Syntax**: `df['col'].astype(type)`
- **Example**:
  ```python
  df['Age'] = df['Age'].astype(int)
  df['Price'] = df['Price'].astype(float)
  ```

#### `.drop_duplicates()`
- **Purpose**: Removes duplicate rows
- **Syntax**: `df.drop_duplicates()`
- **Example**:
  ```python
  df = df.drop_duplicates()
  duplicates_count = df.duplicated().sum()
  ```

---

### 10. **Data Transformation**

#### `.pivot()`
- **Purpose**: Reshapes data (long to wide format)
- **Syntax**: `df.pivot(index='col1', columns='col2', values='col3')`
- **Example**:
  ```python
  pivoted = df.pivot(index='Date', columns='City', values='Temperature')
  ```

#### `pd.melt()`
- **Purpose**: Reshapes data (wide to long format)
- **Syntax**: `pd.melt(df, id_vars=['col'], var_name='name', value_name='value')`
- **Example**:
  ```python
  melted = pd.melt(pivoted.reset_index(), id_vars=['Date'],
                   var_name='City', value_name='Temperature')
  ```

#### Min-Max Scaling
- **Purpose**: Normalizes data to [0, 1] range
- **Formula**: `(x - min) / (max - min)`
- **Example**:
  ```python
  scaled = (df - df.min()) / (df.max() - df.min())
  ```

---

### 11. **Categorical Data Encoding**

#### Ordinal/Label Encoding
- **Purpose**: Maps categories to integers
- **Method**: Using `.map()`
- **Example**:
  ```python
  mapping = {'Low': 1, 'Medium': 2, 'High': 3}
  df['Category_Ordinal'] = df['Category'].map(mapping)
  ```

#### `pd.get_dummies()` (One-Hot Encoding)
- **Purpose**: Converts categorical to binary columns
- **Syntax**: `pd.get_dummies(df['col'], prefix='prefix')`
- **Example**:
  ```python
  encoded = pd.get_dummies(df['Embarked'], prefix='Embarked')
  df = pd.concat([df, encoded], axis=1)
  df = df.drop(columns=['Embarked'])
  ```

---

### 12. **Merging and Joining**

#### `pd.concat()`
- **Purpose**: Concatenates DataFrames
- **Syntax**: `pd.concat([df1, df2], axis=0/1)`
- **Examples**:
  ```python
  pd.concat([df1, df2], axis=0)  # Vertical (rows)
  pd.concat([df1, df2], axis=1)  # Horizontal (columns)
  ```

#### `pd.merge()`
- **Purpose**: SQL-style joins
- **Syntax**: `pd.merge(df1, df2, on='key', how='inner/left/right/outer')`
- **Join Types**:
  - `inner`: Only matching rows
  - `left`: All from left, matching from right
  - `right`: All from right, matching from left
  - `outer`: All rows from both
- **Examples**:
  ```python
  pd.merge(df1, df2, on='ID', how='inner')
  pd.merge(df1, df2, on='ID', how='left')
  pd.merge(df1, df2, on='ID', how='outer')
  ```

---

### 13. **Grouping and Aggregation**

#### `.groupby()`
- **Purpose**: Groups data for aggregation
- **Syntax**: `df.groupby('col').agg_func()`
- **Examples**:
  ```python
  df.groupby('Sex')['Survived'].mean()
  df.groupby(['Sex', 'Embarked'])['Survived'].mean()
  df.groupby('Pclass')['Fare'].agg(['mean', 'median', 'max', 'min'])
  ```

#### `.unstack()`
- **Purpose**: Pivots grouped data
- **Syntax**: `grouped.unstack()`
- **Example**:
  ```python
  survival_by_sex_embarked = df.groupby(['Sex', 'Embarked'])['Survived'].mean().unstack()
  ```

---

### 14. **Advanced Selection**

#### `.select_dtypes()`
- **Purpose**: Selects columns by data type
- **Syntax**: `df.select_dtypes(include/exclude=['type'])`
- **Examples**:
  ```python
  df.select_dtypes(include=['object'])      # Only object columns
  df.select_dtypes(exclude=['object'])      # Exclude object columns
  df.select_dtypes(include=['int64', 'float64'])  # Numeric only
  ```

#### `.nunique()`
- **Purpose**: Counts unique values
- **Syntax**: `df['col'].nunique()`
- **Example**: `df['Sex'].nunique()  # Returns 2`

#### `.unique()`
- **Purpose**: Returns unique values
- **Syntax**: `df['col'].unique()`
- **Example**: `df['Embarked'].unique()  # ['C', 'Q', 'S']`

---

### 15. **Data Binning**

#### `pd.qcut()`
- **Purpose**: Quantile-based discretization
- **Syntax**: `pd.qcut(df['col'], q=n, labels=labels)`
- **Example**:
  ```python
  df['Age_Group'] = pd.qcut(df['Age'], q=5, 
                            labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
  ```

---

### 16. **Working with External Libraries**

#### Loading from sklearn
- **Purpose**: Load built-in datasets
- **Example**:
  ```python
  from sklearn.datasets import load_iris
  iris = load_iris()
  iris_df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
  ```

---

## Quick Reference

### NumPy Key Concepts
- **Vectorization**: Operations on entire arrays at once
- **Broadcasting**: Automatic shape adjustment for operations
- **Memory efficiency**: Contiguous memory storage
- **0-indexed**: Array indexing starts at 0
- **Axis 0**: Rows (vertical)
- **Axis 1**: Columns (horizontal)

### Pandas Key Concepts
- **Series**: 1D labeled array
- **DataFrame**: 2D labeled table
- **Index**: Row labels
- **Columns**: Column labels
- **Inplace operations**: `inplace=True` modifies original, `False` returns copy
- **Method chaining**: `df.drop().fillna().rename()`

### Common Patterns

#### NumPy Performance
```python
import time
# Always faster than Python lists for:
# - Element-wise operations
# - Matrix multiplication
# - Mathematical functions
```

#### Pandas Data Pipeline
```python
# 1. Load data
df = pd.read_csv('file.csv')

# 2. Inspect
df.head()
df.info()
df.describe()

# 3. Clean
df = df.drop_duplicates()
df = df.dropna(subset=['important_col'])
df['col'] = df['col'].fillna(df['col'].mean())

# 4. Transform
df['new_col'] = df['col1'] * df['col2']
encoded = pd.get_dummies(df['category'])
df = pd.concat([df, encoded], axis=1)

# 5. Analyze
grouped = df.groupby('category')['value'].mean()

# 6. Export
df.to_csv('cleaned.csv', index=False)
```

### Important Distinctions

#### NumPy
- `arange(start, stop, step)` vs `linspace(start, stop, num)`
- `array * 2` (element-wise) vs `np.dot(arr1, arr2)` (matrix mult)
- `np.copy()` creates new array, slicing creates view

#### Pandas
- `df['col']` (Series) vs `df[['col']]` (DataFrame)
- `.loc[]` (labels) vs `.iloc[]` (positions)
- `df.groupby()` returns GroupBy object, need aggregation
- `.fillna()` vs `.dropna()` for missing values
- Inner/Left/Right/Outer joins in `pd.merge()`

### Memory Tips
- Use `inplace=True` to avoid creating copies
- Use `.values` to get underlying NumPy array
- Use `dtype` parameter to reduce memory (int8 vs int64)

---

## Exam Tips

1. **Know the difference between**:
   - NumPy arrays vs Python lists
   - Series vs DataFrame
   - `.loc[]` vs `.iloc[]`
   - `concat()` vs `merge()`

2. **Common gotchas**:
   - Array dimensions must match for operations
   - Matrix multiplication is NOT commutative (AB ≠ BA)
   - Default `head()` is 5 rows, `linspace()` is 50 points
   - `inplace=True` returns None

3. **Must-know formulas**:
   - Min-Max Scaling: `(x - min) / (max - min)`
   - Dot product: Sum of element-wise products
   - Matrix inverse: `A @ A_inv = I`
   - Transpose property: `(AB)^T = B^T @ A^T`

4. **Practice explaining**:
   - Why NumPy is faster than Python lists
   - When to use `fillna()` vs `dropna()`
   - Different types of joins and when to use each
   - Purpose of one-hot encoding

Good luck with your viva! 🎓
