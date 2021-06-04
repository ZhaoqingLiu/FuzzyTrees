# FuzzyTrees

FuzzyTrees is a framework designed for rapidly developing various fuzzy decision tree algorithms.

First, the framework is a supporting architecture for development. Based on this framework, any developer can extend more components according to a particular algorithm to quickly build a complete fuzzy decision tree scheme.

Second, the framework provides protocols for extending components. You can follow a unified set of APIs to develop algorithms that are easy for other developers to understand.
To easily extend the components, the framework has provided you with a set of supporting and easy-to-use utilities, such as the split metric calculation and split method tools used in ID3, C4.5, and CART algorithms, respectively.

Also, a fuzzy CART algorithm has implemented based on this framework.

Fuzzytrees是一个框架，它为快速开发各种模糊决策树算法而设计。

首先，该框架是一个用于开发的支撑架构。在该框架的基础上，任何开发者都可以根据某个特定的算法去扩展更多的组成部分，从而迅速地构建一个完整的模糊决策树方案。

其次，该框架提供扩展组件的协议。你可遵循一组统一的应用程序接口开发出易于其他开发者理解的算法。为了方便地扩展组件，该框架已经给你提供了一组辅助性、支撑性的方便易用的实用工具，例如分别在ID3, C4.5, 和CART算法中使用的分裂指标计算和分裂方法的工具。

此外，Fuzzytrees已经在该框架基础上实现了一个模糊CART算法。


## Usage

###  Getting it
```shell
$ pip install fuzzytrees
```

###  Importing dependencies
```shell
$ pip install -r requirements.txt
```

### Using it
```python
from fuzzytrees.fuzzy_decision_tree_wrapper import *
```



License
----

MIT License

Copyright (c) 2021 Zhaoqing Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Contact us: Geo.Liu@outlook.com


