# Udacity Data Scientist Project
## StackOverflow Survey Salary Analysis 2015-2019

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

- colorcet
- [HoloViews](http://holoviews.org/index.html)
- [hvPlot](https://hvplot.holoviz.org/index.html)
- Matplotlib
- numpy
- Pandas
- Bokeh
- [Wordcloud](https://amueller.github.io/word_cloud/index.html)
- SciPy
- SciKit-Learn
- Jupyter Notebook
- The code should run with no issues using Python versions 3.*.

## Project Motivation<a name="motivation"></a>

I wanted to get a deeper understanding in developers' economic return around the world, using Stack Overflow data over the years to understand:

1. Which Countries are best for developers in terms of compensation?
2. Which Countries show an higher than average compensation for developers?
3. What are the main differences between the richest and the poorest Countries?

## File Descriptions <a name="files"></a>

The full analysis for years 2015-2019 is contained in the jupyter notebook. Most of the functions are written in a separate file that needs to be imported named 'AidFunctions.py'.
The World Bank Data and the question schemas are attached, but due to limit in size, the Stack Overflow results have not been uploaded and can be found [here](https://insights.stackoverflow.com/survey).
For the import of these csv, the files should be contained in the 'Project Files' folder, whereas the following naming convention for each year results was used:
- Project Files\<year> Stack Overflow Survey Results

  E.g. 'Project Files\2019 Stack Overflow Survey Results.csv'

## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](https://medium.com/@iamjuststen/this-is-why-becoming-a-developer-pays-out-d733afe044a2).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Credit to Stack Overflow for the data and [DavidSalazarv95](https://www.kaggle.com/davidsalazarv95/average-income-worldwide) for the average incomes dataset.  .
