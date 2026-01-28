---
title: "Closures and Partial Function Application in Python: A NLP Use Case"
pubDate: 2021-11-06T02:53:16+02:00
description: "This article explores closures and partial function application in Python, demonstrating how these powerful concepts can simplify state management and encapsulation. Through practical examples from NLP pipelines, including SpaCy and skweak, you'll learn common pitfalls, effective solutions, and best practices to write cleaner, more maintainable Python code."
category: "lab"
math: true
draft: true
---

While building some pipelines for automatic text annotation at work, I ran into a bug that completely threw me off. It made me realise I didn’t actually understand how Python closures behaved in a `for` loop. The catch was subtle: in Python, a `for` loop doesn’t create a new scope, and that detail can change the outcome of your code in surprising ways.

This wasn’t an academic exercise. We were integrating multiple NLP packages into our annotation workflow, including [SpaCy](https://spacy.io/) and [skweak](https://github.com/NorskRegnesentral/skweak). Once a new model or heuristic was ready, we’d hook it up to label customer complaint text and other incoming feedback. The pipelines were a mix of our own code and library components we couldn’t modify, which made isolating behaviour and managing state pretty important.

That’s where closures became useful:

1. **Encapsulating state**
   A closure can hold on to variables from its outer function, keeping them alive without exposing them globally. This made it easy to keep small, private bits of state for each annotation function without bleeding into the rest of the pipeline.

2. **Keeping it Pythonic**
   Instead of introducing yet another class just to carry some parameters, a closure gave us a clear, lightweight way to add custom logic.

3. **Reducing complexity**
   In a large multi-package system, every small simplification counts. Closures let us wrap up behaviour in neat, self-contained units without needing to rework or fully understand the architecture of the external package we were plugging into.

In this post, I’ll share the exact scenario where closures and partial functions turned out to be the cleanest solution. We’ll rebuild parts of SpaCy and skweak to make the example concrete, then see how these patterns can simplify state management in real NLP workflows.

## What Are Closures?

Before we dive into the problem I encountered, let's briefly talk about what closures are. A closure in Python is a function object that has access to variables in its local scope even after the function has finished execution. This allows for data to be hidden from the global scope, making it possible to encapsulate logic and state within a function.

Here's a simple example:


```python
def outer_function(x):
    def inner_function(y):
        return x + y
    return inner_function

x = -2  # this won't be seen by the inner_function, since it already sees the x from outer_function
add_five = outer_function(x=5)
result = add_five(y=3)
print(result)  # Output will be 8

```

    8


## SpaCy objects

[SpaCy](https://spacy.io/) is here for you to help you build easy NLP pypelines.
Central to this package are the `Doc` objects (short for document). It's a neatly way to pack data for NLP and if it doesn't provide what you need out of the box, you can [always extend it's functionalities](https://spacy.io/usage/processing-pipelines#custom-components-attributes) to match your usecase.


```python
from typing import Callable, Generator, Iterable, List

class Doc:
    def __init__(self, text) -> None:
        self.text = text
        self.tokens: List[str] = text.split()
        self.spans: Span = []
    def __len__(self):
        return len(self.tokens)
    def __getitem__(self, position):
        return self.tokens[position]
```

By implementing `__len__` and `__getitem__` [double-under functions](https://www.geeksforgeeks.org/dunder-magic-methods-python/) we got the ability to iterate through the Doc's tokens with a simple for as below. This is thanks to the Python datamodel. It's outside the scope of this post, but learning to leverage the datamodel will pay dividends on your effectiveness in Python. [Fluent Python](https://learning.oreilly.com/library/view/fluent-python/9781491946237/) introduces it in the first chapter in a very neat way. If you like video format more, [James Powell](https://www.youtube.com/watch?v=AmHE0kZhLIQ) got you covered.


```python
doc = Doc("Today I ate garbonzo beans")
for token in doc:
    print(token)
```

    Today
    I
    ate
    garbonzo
    beans


A `Span` is a slice of a `Doc`. Usually it can.. span multiple tokens, but today I have a feeling that all the spans we'll look at will match exactly one token. Also, in our case the spans will be always labeled.


```python
from typing import NamedTuple

class Span(NamedTuple):
    position: int
    label: str

```

# skweak functions
If you haven't looked at the [skweak repo](https://github.com/NorskRegnesentral/skweak) yet, it suffices to know that it provides a neat way of composing simple annotators to get a better one.
Now, skweak provides us with some very interesting classes. One is a `FunctionAnnotator`. This takes a function that returns a list of spans from a document and attaches these spans to the given document. 


```python
class FunctionAnnotator:
    def __init__(self, function: Callable[[Doc], Iterable[Span]]):
        self.find_spans = function
		
    def __call__(self, doc: Doc) -> Doc:
        # We start by clearing all existing annotations
        doc.spans = []

        for position, label in self.find_spans(doc):
            doc.spans.append(Span(position, label))

        return doc
```

Let's see a simple labeling function we may use


```python
def animal_labeling_function(doc: Doc) -> Generator[Span, None, None]:
    for position, token in enumerate(doc.tokens):
        if token.startswith('a'):
            yield Span(position, 'ANIMAL')

doc = Doc('this animal is some kind of antilope')
animal_annotator = FunctionAnnotator(animal_labeling_function)
doc = animal_annotator(doc)

print(doc.spans)
```

    [Span(position=1, label='ANIMAL'), Span(position=6, label='ANIMAL')]


The `FunctionAnnotatorAggregator` takes multiple annotator functions and combines them in some fancy way. We won't do it justice with the implementation below.
We'll just make it force our documents to have a maximum of one label per span. We will also sort them by the order of appearance.


```python
class FunctionAnnotatorAggregator:
    """Base aggregator to combine all labelling sources into a single annotation layer"""
    def __init__(self, annotators: Iterable[FunctionAnnotator]) -> None:
        self.annotators = annotators
    
    def __call__(self, doc: Doc) -> Doc:
        spans_dict = dict()
        for annotator in self.annotators:
            for span in annotator(doc).spans:
                spans_dict[span.position] = span.label
        
        doc.spans = []
        for position, label in spans_dict.items():
            doc.spans.append(Span(position, label))
        doc.spans.sort()
        
        return doc
```


```python
def verb_labeling_function(doc: Doc) -> Generator[Span, None, None]:
    for position, token in enumerate(doc.tokens):
        if token in ['is', 'has']:
            yield Span(position, 'VERB')

verb_annotator = FunctionAnnotator(verb_labeling_function)

aggregated_annotator = FunctionAnnotatorAggregator([animal_annotator, verb_annotator])

doc = aggregated_annotator(doc)

print(doc.spans)

```

    [Span(position=1, label='ANIMAL'), Span(position=2, label='VERB'), Span(position=6, label='ANIMAL')]


# The problem

The packages are well implemented and work as expected! 
Now, we may wish to programatically generate some labeling functions from a list of excellent heuristic parameters
<a id='problem_cell'></a>


```python
heuristic_parameters = [
    ('M', 'MAMMAL'),
    ('F', 'FISH'),
    ('B', 'BIRD')
    ]

labeling_functions = []
for strats_with, label in heuristic_parameters:
    def labeling_function(doc: Doc) -> Generator[Span, None, None]:
        for position, word in enumerate(doc.tokens):
            if word.startswith(strats_with):
                yield Span(position, label)
    labeling_functions += [labeling_function]

strats_with, label = 'B', 'BOVINE'

```

<a id='problem_cell_2'></a>


```python
doc = Doc("Monkeys are red. Firefish are blue; Besra is a bird and so are you")

# we'll define this function since we'll use it a lot more below
def print_spans_from_labelers(doc, labeling_functions):
    annotators = [FunctionAnnotator(labeling_function) for labeling_function in labeling_functions]
    aggregated_annotator = FunctionAnnotatorAggregator(annotators)

    doc = aggregated_annotator(doc)

    print(doc.spans)
print_spans_from_labelers(doc, labeling_functions)

```

    [Span(position=6, label='BOVINE')]


What happened? It seems that only the last function was applied. Let's look at the `labeling_functions`


```python
for labeling_function in labeling_functions:
    print(labeling_function)
```

    <function labeling_function at 0x10372e200>
    <function labeling_function at 0x10372f1c0>
    <function labeling_function at 0x10372f250>


They point to different memory addresses.
Let's rewrite this with lambda functions. 

*Note* if you haven't worked with list comprehensions before: don't worry about it; think of the code below as a way to create a new function without replacing the existing function with the same name


```python
labeling_functions = [
                        lambda doc: 
                            ( # this is also a generator in Python; it's the same syntax as list comprehension
                            # but we use round braces instead of square ones
                                Span(position, label) 
                                    for position, word in enumerate(doc.tokens) 
                                    if word.startswith(strats_with)
                            )
                        for strats_with, label in heuristic_parameters
                    ]

for labeling_function in labeling_functions:
    print(labeling_function)
```

    <function <listcomp>.<lambda> at 0x10372dc60>
    <function <listcomp>.<lambda> at 0x10372f490>
    <function <listcomp>.<lambda> at 0x10372f5b0>


But when we want to print the function the problem stays.


```python
print_spans_from_labelers(doc, labeling_functions)
```

    [Span(position=6, label='BIRD')]


This is because of scoping. The problem is that, since we didn't declare strats_with, label in the lambda body or parameters, the lambdas will always look in the scope immediately outside them and they will find the last values that `strats_with`, `label` had.
If you come from other languages it might be strange to you, but Python doesn't create a new scope or context inside the `for` body. Instead it uses the same local scope. This is why `strats_with, label = 'B', 'BOVINE'` in snippet [8](#problem_cell) produced snippet [9](#problem_cell_2) to display the label as 'BOVINE'




But be not affraid! There is a solution:


```python
def function_closure(strats_with, label):
    local_strats_with, local_label = strats_with, label
    def labeling_function(doc: Doc) -> Generator[Span, None, None]:
        for position, word in enumerate(doc.tokens):
            if word.startswith(local_strats_with):
                yield Span(position, local_label)
    return labeling_function

labeling_functions = [function_closure(strats_with, label) for strats_with, label in heuristic_parameters]
```

Now, when we get the annotators things go as expected.


```python
print_spans_from_labelers(doc, labeling_functions)
```

    [Span(position=0, label='MAMMAL'), Span(position=3, label='FISH'), Span(position=6, label='BIRD')]


But why is this different from the last attempt? This time, by calling `function_closure` we are creating a local scope around the labeling function and put the `strats_with` and `label` variables in it. These variables are recreated every time we call `function_closure`. It also recreates `labeling_function` since functions are regular objects in Python and different calls can’t trample on one another’s local variables. 

A good mental model is to think of function values as containing both the code in their body and the environment in which they are created.*

*lifted from [*Eloquent Javascript*](https://eloquentjavascript.net/03_functions.html) book

Inspecting the functions will also confirm this: 


```python
print(repr(labeling_functions[0]))
print(type(labeling_functions[0]))
```

    <function function_closure.<locals>.labeling_function at 0x10372f640>
    <class 'function'>


One improvement we can make is not using `local_strats_with`, `local_label`, since parameters are themselves local variables 


```python
def function_closure(strats_with, label):
    def labeling_function(doc: Doc) -> Generator[Span, None, None]:
        for position, word in enumerate(doc.tokens):
            if word.startswith(strats_with):
                yield Span(position, label)
    return labeling_function

labeling_functions = [function_closure(strats_with, label) for strats_with, label in heuristic_parameters]
print_spans_from_labelers(doc, labeling_functions)
```

    [Span(position=0, label='MAMMAL'), Span(position=3, label='FISH'), Span(position=6, label='BIRD')]


# Partial function application

Yet another way to do things, partial application is a concept where you fix a few arguments of a function and generate a new function:


```python
from functools import partial

def labeling_function(starts_with, label, doc: Doc) -> Generator[Span, None, None]:
    for position, word in enumerate(doc.tokens):
        if word.startswith(starts_with):
            yield Span(position, label)

# Using partial to fix the first two arguments starts_with and label
labeling_functions = [partial(labeling_function, starts_with, label) for starts_with, label in heuristic_parameters]

# Verify if the partial application works as intended
print_spans_from_labelers(doc, labeling_functions)


```

    [Span(position=0, label='MAMMAL'), Span(position=3, label='FISH'), Span(position=6, label='BIRD')]


Explanation:

`labeling_function` is now a single function that accepts three parameters.

We use `functools.partial` to "lock in" the first two parameters, `starts_with` and `label`.

This generates a new function for each pair of `starts_with` and `label`, which we then add to `labeling_functions`.

Now compare it with how you'd implement a regular class:


```python
class LabelingCallable:
    def __init__(self, starts_with, label) -> None:
        self.starts_with = starts_with
        self.label = label

    def __call__(self, doc) -> Generator[Span, None, None]:
        for position, word in enumerate(doc.tokens):
            if word.startswith(self.starts_with):
                yield Span(position, self.label)

labeling_functions = [LabelingCallable(strats_with, label) for strats_with, label in heuristic_parameters]
print_spans_from_labelers(doc, labeling_functions)
```

    [Span(position=0, label='MAMMAL'), Span(position=3, label='FISH'), Span(position=6, label='BIRD')]


The object is used only because it can function as a regular function - something that an actual regular function is more fit to do. This also requires you to come up with a naming convention for this kind of classes. And it doesn't fit with the fact that `skweak` expects a function (as the code and docstrings imply), even if it masquerades as one.

You can also do something like this:


```python
class LabelingCallable:
    def __init__(self, starts_with, label) -> None:
        self.starts_with = starts_with
        self.label = label

    def labeling_function(self, doc) -> Generator[Span, None, None]:
        for position, word in enumerate(doc.tokens):
            if word.startswith(self.starts_with):
                yield Span(position, self.label)

labeling_functions = [LabelingCallable(strats_with, label).labeling_function for strats_with, label in heuristic_parameters]
print_spans_from_labelers(doc, labeling_functions)
print(repr(labeling_functions[0]))
print(type(labeling_functions[0]))
```

    [Span(position=0, label='MAMMAL'), Span(position=3, label='FISH'), Span(position=6, label='BIRD')]
    <bound method LabelingCallable.labeling_function of <__main__.LabelingCallable object at 0x1037524a0>>
    <class 'method'>


Maybe this is something you'll eventually want but, for our intended purposes, this is basically a closure with extra steps.

# Conclusion

In this post we explored closures, a solution for providing data locality.

As we've seen, Python closures and partial function application are powerful features for encapsulating local state. These techniques can be especially useful in NLP pipelines, allowing us to write clean, modular code. Whether you are working on a simple task or something complex like automated text annotation, understanding these language features can significantly improve your code quality and maintainability.

If this is the kind of content you enjoy, let me know!
