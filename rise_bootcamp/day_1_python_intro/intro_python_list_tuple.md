# Introduction to Python --- List


```python
langs = ['Python', 'Fortran', 'C', 'Julia', 'Java']
projs = ['ProjA', 'ProjB', 'ProjC', 'ProjD', 'ProjE']
print(langs)
print(type(langs))
```

    ['Python', 'Fortran', 'C', 'Julia', 'Java']
    <class 'list'>



```python
str_a = 'Python/C++/Fortran/Julia/Rust'
lst_lang = str_a.split('/')
print(lst_lang)

str_b = 'abcdef'
lst_char = list(str_b)
print(lst_char)
```

    ['Python', 'C++', 'Fortran', 'Julia', 'Rust']
    ['a', 'b', 'c', 'd', 'e', 'f']



```python
# using list comprehension
lst_a = [x for x in range(9)]
print(lst_a)
lst_b = [x for x in range(9) if x%2 == 0]
print(lst_b)
lst_c = [x if x%2 == 0 else -x for x in range(9)]
print(lst_c)
```

    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    [0, 2, 4, 6, 8]
    [0, -1, 2, -3, 4, -5, 6, -7, 8]



```python
# nested list
langs = [
    ['C 89', 'C 99', 'C 11', 'C17'], 
    ['Python 2.7', 'Python 3.6', 'Python 3.12'], 
    ['Fortran 77', 'Fortran 90', 'Fortran 03']
]
print(langs[1])
print(langs[1][1])
print()

langs = ['C', 'Python', 'Fortran']
vers = ['VerA', 'VerB', 'VerC']
lang_ver_1 = [langs, vers]
print(lang_ver_1, end='\n\n')
lang_ver_2 = [[lang, ver] for lang in langs for ver in vers]
print(lang_ver_2, end='\n\n')
lang_ver_3 = list(zip(langs, vers))
print(lang_ver_3)
```

    ['Python 2.7', 'Python 3.6', 'Python 3.12']
    Python 3.6
    
    [['C', 'Python', 'Fortran'], ['VerA', 'VerB', 'VerC']]
    
    [['C', 'VerA'], ['C', 'VerB'], ['C', 'VerC'], ['Python', 'VerA'], ['Python', 'VerB'], ['Python', 'VerC'], ['Fortran', 'VerA'], ['Fortran', 'VerB'], ['Fortran', 'VerC']]
    
    [('C', 'VerA'), ('Python', 'VerB'), ('Fortran', 'VerC')]


## Methods for List


```python
# slicing
lst_a = [x for x in range(20) if x%2 == 1]
print(lst_a)
print(lst_a[1:6])
print(lst_a[5:])
```

    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    [3, 5, 7, 9, 11]
    [11, 13, 15, 17, 19]



```python
##### append, insert, pop, remove
lst_a.append(21)
print(lst_a)
lst_a.append(23)
print(lst_a)
lst_a.insert(0, -1)
print(lst_a)
lst_a.insert(5, 100)
print(lst_a, end='\n\n')

lst_a.pop()
print(lst_a)
lst_a.pop()
print(lst_a)
lst_a.pop(0) # not recommend (low efficiency)
print(lst_a, end='\n\n')

lst_a.remove(100) # remove one element from the list
print(lst_a)
lst_a.remove(11) # remove one element from the list
print(lst_a, end='\n\n')
# lst_a.remove(105) # gets a `ValueError` as the element `105` is not available, 

del lst_a[3] # delete the third element in the list
print(lst_a)
del lst_a[5] # delete the third element in the list
print(lst_a)
# del lst_a[10] # IndexError: list assignment index out of range

```

    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    [-1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    [-1, 1, 3, 5, 7, 100, 9, 11, 13, 15, 17, 19, 21, 23]
    
    [-1, 1, 3, 5, 7, 100, 9, 11, 13, 15, 17, 19, 21]
    [-1, 1, 3, 5, 7, 100, 9, 11, 13, 15, 17, 19]
    [1, 3, 5, 7, 100, 9, 11, 13, 15, 17, 19]
    
    [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    [1, 3, 5, 7, 9, 13, 15, 17, 19]
    
    [1, 3, 5, 9, 13, 15, 17, 19]
    [1, 3, 5, 9, 13, 17, 19]



```python
lst_a.append([25, 27, 29]) # now lst_a is a nested list
print(lst_a, end='\n\n')

lst_a = [x for x in range(10) if x%2 == 1]
lst_b = [x for x in range(10, 20) if x%2 == 0]
print(lst_a, lst_b, sep='  xxxxx ')
lst_c = lst_a + lst_b
print(lst_c, end='\n\n')

# list.extend(list) vs list.append(list) vs list.append(item)
lst_c.extend(lst_b) # now lst_c is a list
print(lst_c)
lst_c.append(lst_b) # now lst_c is a nested list
print(lst_c)
```

    [1, 3, 5, 9, 13, 17, 19, [25, 27, 29]]
    
    [1, 3, 5, 7, 9]  xxxxx [10, 12, 14, 16, 18]
    [1, 3, 5, 7, 9, 10, 12, 14, 16, 18]
    
    [1, 3, 5, 7, 9, 10, 12, 14, 16, 18, 10, 12, 14, 16, 18]
    [1, 3, 5, 7, 9, 10, 12, 14, 16, 18, 10, 12, 14, 16, 18, [10, 12, 14, 16, 18]]


### indexing is similar to that for strings


```python
# sorting lists

lst_a = [2, 0, 9, 1, 6, 3, 8, 7]
lst_a.sort()
print(lst_a, end='\n\n')

lst_a = [2, 0, 9, 1, 6, 3, 8, 7]
lst_b = sorted(lst_a)
print(lst_b)
print(lst_a)
```

    [0, 1, 2, 3, 6, 7, 8, 9]
    
    [0, 1, 2, 3, 6, 7, 8, 9]
    [2, 0, 9, 1, 6, 3, 8, 7]



```python
lst_a = [2, 0, 9, 1, 6, 3, 8, 7]
lst_a.reverse()
print(lst_a)
lst_a.reverse()
print(lst_a)
```

    [7, 8, 3, 6, 1, 9, 0, 2]
    [2, 0, 9, 1, 6, 3, 8, 7]



```python
# shallow copy vs deep copy

lst_a = [1, 5, [7, 11]]
lst_b = lst_a
lst_c = lst_a[:]
print("ID of lst_a = ", id(lst_a)) # `id()` to get the physical address in the memory
print("ID of lst_b = ", id(lst_b))
print("ID of lst_c = ", id(lst_c))
print()

lst_a.append(9)
print("After append 9, lst_a =", lst_a)
print("After append 9, lst_b =", lst_b)
print("After append 9, lst_c =", lst_c)
print()

lst_a[2].append(13)
print("After append 13 in sublist, lst_a =", lst_a)
print("After append 13 in sublist, lst_b =", lst_b)
print("After append 13 in sublist, lst_c =", lst_c)

```

    ID of lst_a =  21788247744
    ID of lst_b =  21788247744
    ID of lst_c =  21788252416
    
    After append 9, lst_a = [1, 5, [7, 11], 9]
    After append 9, lst_b = [1, 5, [7, 11], 9]
    After append 9, lst_c = [1, 5, [7, 11]]
    
    After append 13 in sublist, lst_a = [1, 5, [7, 11, 13], 9]
    After append 13 in sublist, lst_b = [1, 5, [7, 11, 13], 9]
    After append 13 in sublist, lst_c = [1, 5, [7, 11, 13]]


# Introduction to Python --- Tuple


```python
tuple_a = () # an empty tuple
print(type(tuple_a))
```

    <class 'tuple'>



```python
tuple_a = ('a', 2, 3.14, "hello")
print(type(tuple_a))
tuple_b = 'a', 2, 3.14, "hello"
print(type(tuple_b))
```

    <class 'tuple'>
    <class 'tuple'>



```python
tuple_a = (2)  # this is an integer, not a tuple
tuple_b = (2,) # if there is only one element in the tuple, use a `,` after the element
print(type(tuple_a))
print(type(tuple_b))
```

    <class 'int'>
    <class 'tuple'>



```python
print(3*(11+12))  # this is a simple arithmetic calculation
print(3*(11+12,)) # this is to create a tuple consisting of three elements
```

    69
    (23, 23, 23)



```python
# the build-in function `tuple()` can transform lists and strings to a tuple
lst = [1,2,3,4,5]
tuple_lst = tuple(lst)
print(tuple_lst)
print(type(tuple_lst), end='\n\n')

string = "hello, python!"
tuple_string = tuple(string)
print(tuple_string)
print(type(tuple_string))
```

    (1, 2, 3, 4, 5)
    <class 'tuple'>
    
    ('h', 'e', 'l', 'l', 'o', ',', ' ', 'p', 'y', 't', 'h', 'o', 'n', '!')
    <class 'tuple'>



```python
# nested structures

lst_a = [1,2,3]; lst_b = [4,5,6]
tuple_a = lst_a, lst_b
print(tuple_a)

tuple_b = 7,8,9
tuple_c = lst_a + lst_b, tuple_b
print(tuple_c)
```

    ([1, 2, 3], [4, 5, 6])
    ([1, 2, 3, 4, 5, 6], (7, 8, 9))


## List vs Tuple


```python
# tuple is immutable, list is mutable (`append`, `remove`, `del`, `pop`, `insert`, etc.)

tuple_a = (1,4,6,7,8,9)
print(tuple_a[2])
# tuple_a[2] = 11 # TypeError: 'tuple' object does not support item assignment

# if you want to place one elemnt in the tuple with a new one, you can use `tuple -> list -> tuple`
lst_tuple = list(tuple_a)
lst_tuple[2] = 11
tuple_b = tuple(lst_tuple)
print(tuple_b)

# check the physical address for these two tuples
print(id(tuple_a), id(tuple_b), sep='  xxx  ')
```

    6
    (1, 4, 11, 7, 8, 9)
    21801556032  xxx  21801556224



```python
# you cannot change/remove/add elements in the tuple,
#    but if there is a mutable item (such as a list) in the tuple, you can change/add/remove lelements in the list
tuple_a = (1, 2.3, 'a', [4, 7, 9, 'b'], "hello python")
print(tuple_a)
print(tuple_a[3])
tuple_a[3].pop()
print(tuple_a)
tuple_a[3].insert(1, "hello, C")
print(tuple_a)
```

    (1, 2.3, 'a', [4, 7, 9, 'b'], 'hello python')
    [4, 7, 9, 'b']
    (1, 2.3, 'a', [4, 7, 9], 'hello python')
    (1, 2.3, 'a', [4, 'hello, C', 7, 9], 'hello python')



```python
# check the memory usage using the `getsizeof` function in the `sys` module
import sys
lst_a = list(range(100000000))
tuple_a = tuple(range(100000000))
range_obj = range(100000000) # this is `range` object
print(sys.getsizeof(lst_a))
print(sys.getsizeof(tuple_a))
print(type(range_obj), sys.getsizeof(range_obj), end='\n\n')

# check their efficieies using the `timeit` function in the `timeit` module
import timeit
print(timeit.timeit('[1, 2, 3, 4, 5, 6, 7, 8, 9]'))
print(timeit.timeit('(1, 2, 3, 4, 5, 6, 7, 8, 9)'))
```

    800000056
    800000040
    <class 'range'> 48
    
    0.024579750024713576
    0.004987250023987144



```python
tuple_a = tuple(range(8))
print(tuple_a)
tuple_b = tuple_a[1:4] + tuple_a[6:9]
print(tuple_b)

# tuple_c = tuple_a[2:5] + tuple_a[7] # gets an error as `tuple_a[7]` is not a tuple
tuple_c = tuple_a[2:5]
tuple_c += tuple_a[7],
print(tuple_c, end='\n\n')

# `tuple_a[7]` is an element in the tuple, and `tuple_a[7],` is a tuple even there in only one element in this tuple
temp_a = tuple_a[7]
temp_b = tuple_a[7],
print(temp_a, type(temp_a))
print(temp_b, type(temp_b))
```

    (0, 1, 2, 3, 4, 5, 6, 7)
    (1, 2, 3, 6, 7)
    (2, 3, 4, 7)
    
    7 <class 'int'>
    (7,) <class 'tuple'>


## packing and unpacking


```python
# swap the value of two variables
a = 11
b = 23
a, b = b, a # a full expression is (a, b) = (b, a)
print(a, b)
```

    23 11



```python
year_a = 2023
lang_a = 'python'
proj_a = 'bootcamp'
info = year_a, lang_a, proj_a # this is packing
print(info[0], info[1], info[2], end='\n\n')

year_b, lang_b, proj_b = info # this is unpacking
print(year_b, lang_b, proj_b)

# year_b, lang_b, proj_b, temp_a = info # ValueError: not enough values to unpack (expected 4, got 3)
# year_b, lang_b = info                 # ValueError: too many values to unpack (expected 2)
```

    2023 python bootcamp
    
    2023 python bootcamp



```python
# a solution for this `ValueError` is to use `*` during unpacking
tuple_a = 1, 10, 100, 1000

i, j, *k = tuple_a
print(i, j, k)

i, *j, k = tuple_a
print(i, j, k)

*i, j, k = tuple_a
print(i, j, k, end='\n\n')

i, *j = tuple_a
print(i, j)

*i, j = tuple_a
print(i, j, end='\n\n')

i, j, k, *l = tuple_a
print(i, j, k, l, end='\n\n')
print(type(i), type(j), type(k), type(l)) # i, j, k are integers, and l is a list containing one element

i, j, k, l, *m = tuple_a
print(i, j, k, l, m)
print(type(l), type(m)) # l is an integer, and m is an empty list
```

    1 10 [100, 1000]
    1 [10, 100] 1000
    [1, 10] 100 1000
    
    1 [10, 100, 1000]
    [1, 10, 100] 1000
    
    1 10 100 [1000]
    
    <class 'int'> <class 'int'> <class 'int'> <class 'list'>
    1 10 100 1000 []
    <class 'int'> <class 'list'>



```python
# using `dir()` to get all available methods
dir(tuple) # for tuple, it only has two methods, `count` and `index`
```




    ['__add__',
     '__class__',
     '__class_getitem__',
     '__contains__',
     '__delattr__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getitem__',
     '__getnewargs__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__iter__',
     '__le__',
     '__len__',
     '__lt__',
     '__mul__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__rmul__',
     '__setattr__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     'count',
     'index']




```python

```
