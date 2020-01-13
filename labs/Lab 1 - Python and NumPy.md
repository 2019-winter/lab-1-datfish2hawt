---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Name(s)
**PUT YOUR FULL NAME(S) HERE**


**Instructions:** This is an individual assignment, but you may discuss your code with your neighbors.


# Python and NumPy

While other IDEs exist for Python development and for data science related activities, one of the most popular environments is Jupyter Notebooks.

This lab is not intended to teach you everything you will use in this course. Instead, it is designed to give you exposure to some critical components from NumPy that we will rely upon routinely.

## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)

**In the space provided below, what are three things that still remain unclear or need further explanation?**


**YOUR ANSWER HERE**


## Exercises 1-7
For the following exercises please read the Python appendix in the Marsland textbook and answer problems A.1-A.7 in the space provided below.


## Exercise 1

```python

import numpy as np
a = 2 * np.ones((6,4))
a
```

## Exercise 2

```python
b = np.ones((6,4)) + 2 * np.vstack([np.eye(4), np.zeros((2,4))])
b
```

## Exercise 3


For the multiplication operation, if C = A * B, then C[i,j] = A[i,j] * B[i,j] i.e. element-wise multiplication

Thus A * B will only work for matrices of the same size

However the dot operation is a matrix multiplication, so for dot(A, B) to work,

The number of columns in A must be equal to the number of rows in B


## Exercise 4


Let a be an m0 x n0 matrix and b be a m1 x n1 matrix

If n0 = m1 the matrices can be multiplied

The resulting matrix will be of size m0 x n1


## Exercise 5

```python
def print_to_screen():
    print("Hello world!")
    
print_to_screen()
```

## Exercise 6

```python
def random_info():
    a = np.random.rand(3)
    b = np.random.randn(3)
    c = a + b
    print("size: " + str(np.size(c)))
    print("sum: " + str(np.sum(c)))
    print("mean: " + str(np.mean(c)))
    print("standard deviation: " + str(np.std(c)))
          
random_info()
```

## Exercise 7

```python
def howmanyones(array):
    ones = 0
    for el in np.nditer(array):
        if el == 2:
            ones += 1
    return ones

def oneswhere(array):
    a = np.where(array == 1, array, 0)
    return np.sum(a)

howmanyones(np.random.randint(-1, 8, (10, 10)))
oneswhere(np.random.randint(-1, 8, (10, 10)))
    
```

## Excercises 8-???
While the Marsland book avoids using another popular package called Pandas, we will use it at times throughout this course. Please read and study [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) before proceeding to any of the exercises below.


## Exercise 8
Repeat exercise A.1 from Marsland, but create a Pandas DataFrame instead of a NumPy array.

```python
import pandas as pd
import numpy as np

pd.DataFrame(2 * np.ones((6,4)))
```

## Exercise 9
Repeat exercise A.2 using a DataFrame instead.

```python
pd.DataFrame(np.ones((6,4)) + 2 * np.vstack([np.eye(4), np.zeros((2,4))]))
```

## Exercise 10
Repeat exercise A.3 using DataFrames instead.


When using DataFrames, column names must match the index names


## Exercise 11
Repeat exercise A.7 using a dataframe.

```python
a = pd.DataFrame((np.random.randint(-1,8,(10,10))))
def hwmny1s():
    count = 0
    for i in range(len(a)):
        for j in a.loc[i]:
            if j == 1:
                count += 1
    return count

def whereda1s():
    return int(sum(np.array(a.where(a == 1).sum())))

hwmny1s()
whereda1s()
```

## Exercises 12-14
Now let's look at a real dataset, and talk about ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df
```

Notice how we have nice headers and mixed datatypes? That is one of the reasons we might use Pandas. Please refresh your memory by looking at the 10 minutes to Pandas again, but then answer the following.


## Exercise 12
How do you select the ``name`` column without using .iloc?

```python
titanic_df["name"]
```

## Exercise 13
After setting the index to ``sex``, how do you select all passengers that are ``female``? And how many female passengers are there?

```python
titanic_df.set_index('sex',inplace=True)
titanic_df.loc['female']
```

## Exercise 14
How do you reset the index?

```python
titanic_df.reset_index(inplace=True)
```

```python
titanic_df
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```
