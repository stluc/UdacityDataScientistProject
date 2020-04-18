import numpy as np
import pandas as pd
import os.path
import bokeh
from bokeh.models import PanTool, WheelZoomTool, Axis, HoverTool
import colorcet as cc
import holoviews as hv
from holoviews import opts, dim, Cycle
import hvplot.pandas
from hvplot import hvPlot
import matplotlib.pyplot as plt
from scipy import interpolate
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def StackOverflow_dataset_import(folder, year):
    '''
    Arguments:
        folder - string of subfolder name where script is stored
        year - year to be searched
    
    import file example (variable = year) -> "2016 Stack Overflow Survey Results.csv"
    import file example (variable = year) -> "2016 Stack Overflow Survey Schema.csv"
    '''
    df = pd.DataFrame()
    schema = pd.DataFrame()
    try:
        df_pattern = str(year)+" Stack Overflow Survey Results.csv"
        schema_pattern = str(year)+" Stack Overflow Survey Schema.csv"
        df_path = os.path.join(".",folder, df_pattern)
        if os.path.isfile(df_path):
            df = pd.read_csv(df_path, low_memory=False)
            print(df_pattern+' successfully loaded.')
            schema_path = os.path.join(".",folder, schema_pattern)
            if os.path.isfile(schema_path):
                schema = pd.read_csv(schema_path, low_memory=False)
                print(schema_pattern+' successfully loaded.')
            else:
                schema = pd.DataFrame()
                print(schema_pattern+' not found.')
        else:
            df = pd.DataFrame()
            schema = pd.DataFrame()
            print("Dataset not found.")
    except:
        print("Wrong dataset format.")
    return df, schema


def clean_no_response(df, response, how='any'):
    '''
    Removes from DataFrame the rows which are missing from the desired response column, along with all completely missing rows or columns.
    Arguments: 
        df: dictionary containing the dataframes, keys are years
        response: string containing the name of the response column
        how (optional): method to pass to select which indexes to drop. Valid 'any' or 'all'. Default is 'any'.
    '''
    df_new = {}
    for year in df.keys():
        try:
            # drop missing values
            df_new[year] = df[year].dropna(subset=response, axis=0, how=how)
            # drop empty columns
            df_new[year] = df_new[year].dropna(how='all', axis=1)
            # drop empty rows
            df_new[year] = df_new[year].dropna(how='all', axis=0)
        except:
            df_new[year] = df[year]
            print(year, "not replaced.")
            continue
    return df_new

def return_choice(string, sub):
    '''
    INPUT
        string - a string to verify if it contains the substring
        sub - tuple where 1st element is substring to verify
    OUTPUT
        return 1 if present
        return 0 if absent or incompatible string
    '''
    if type(string) == str:
        if sub in string:
            return 1
        else:
            return 0
    else:
        return 0

def replace_wrong_answers(series, rep, sep=';'):
    '''
    INPUT
        series - Pandas series to be searched for multiple choices separated by 'sep'
        rep - string for replacement where multiple choices are met
        sep (optional) - string separator btw. multiple choices
    OUTPUT
        new_series - Pandas series with replaced text
    '''
    if type(rep) == str:
        wrong_gender = series.dropna().apply(lambda x: True if ';' in x else False)
        wrong_gender = wrong_gender[wrong_gender==True]
        series.loc[wrong_gender.index] = rep
        return series


def clean_2019_data(df):
    '''
    Basic cleaning for some columns in 2019 data.
    WorkWeekHrs false values, difficult to interpret multiple choices and seminumeric columns
    INPUT
        df - dataframe of StackOverflow 2019 data
    OUTPUT
        df - dataframe properly cleaned
    '''
    if 'WorkWeekHrs' in df.columns:
        # we fix WorkWeekHrs values > 84 (14Hrs/6 days) by dividing by 10
        replace_values = df[df['WorkWeekHrs'] > 84]['WorkWeekHrs']
        df.loc[replace_values.index,'WorkWeekHrs'] = replace_values.values/10
        # drop outlier rows in WorkWeekHrs (values > 112, 16Hrs / 7 days)
        df.drop(df[df['WorkWeekHrs'] > 112].index, inplace=True)
    
    #We now want to fix the incompatible answers in the Gender, Sexuality, Ethnicity columns.
    # Since only one answer can be correct in these fields
    # We assume for Gender that any multiple answer can be classified as non-binary, 
    # for Sexuality that any multiple answer can be classified as Bisexual,
    # for Ethnicity that any multiple answer can be classified as Multiracial.
    repl_dict = {'Gender':'Non-binary, genderqueer, or gender non-conforming',
                 'Sexuality':'Bisexual', 'Ethnicity':'Multiracial'}
    for col, rep in repl_dict.items():
        if col in df.columns:
            df[col] = replace_wrong_answers(df[col], rep)
        
    # We also want to fix the columns which should contain numbers but have a couple of strings.
    # These are 'YearsCode', 'Age1stCode', 'YearsCodePro'.
    repl_dict = {'Less than 1 year':'0','More than 50 years':'55','Younger than 5 years':'4'}
    semi_numeric_cols = ['YearsCode', 'Age1stCode', 'YearsCodePro']
    for col in semi_numeric_cols:
        if col in df.columns:
            df[col] = df[col].replace(repl_dict).astype('float16')

    return df


def encode_categories(data, encoder):
    '''
    Encode non-null data and replace it in the original data
    INPUT
        data - dataframe to be encoded
        encoder - encoder to be used, tested with ordinal encoder
    OUTPUT
        data - dataframe encoded
        revert_dict - a dictionary which can be used for reversing the values
    '''
    #retains only non-null values
    nonulls = np.array(data.dropna())
    #reshapes the data for encoding
    impute_reshape = nonulls.reshape(-1,1)
    #encode date
    impute_ordinal = encoder.fit_transform(impute_reshape)
    a = impute_reshape.transpose().tolist()[0]
    b = impute_ordinal.transpose().tolist()[0]
    revert_dict = dict(zip(b, a))
    #Assign back encoded values to non-null values
    data.loc[data.notnull()] = np.squeeze(impute_ordinal)
    data = data.astype('float16')
    return data, revert_dict


def encode_multiple_choices(df):
    '''
    Introduce multiple columns for each seaparate multiple choice.
    For these columns a missing answer results in 0 in all columns.
    Multiple choice columns are replaced and then removed.
    INPUT
        df - dataframe containing the multiple choice columns
    OUTPUT
        df - one-hot encoded for each of the multiple choices
    '''
    mult_choice_cols = ['EduOther', 'JobFactors', 'DevType', 'LanguageWorkedWith', 'DatabaseWorkedWith', 
                        'PlatformWorkedWith', 'WebFrameWorkedWith', 'MiscTechWorkedWith', 'DevEnviron', 
                        'Containers']
    mult_choice_dict = {}    
    for col in mult_choice_cols:
        if col in df.columns:
            mult_choice_dict[col] = sorted(list(set(df[col].str.cat(sep=';').split(';'))))

    # we build the new columns with 1/0 values to replace the categoricals
    for idx, col in enumerate(mult_choice_cols):
        if col in df.columns:
            for choice in mult_choice_dict[col]:
                new_col = mult_choice_cols[idx] + '_' + choice
                df[new_col] = df[mult_choice_cols[idx]] \
                                            .apply(return_choice, args=(choice,)).astype('int8')

    # we can now drop the original columns
    df.drop(mult_choice_cols, axis=1, inplace=True, errors='ignore')
    return df


def KNN_impute_dataframe(df, n_rows):
    '''
    Impute numerical missing values from dataframe. Splits the dataset to reduce the burden on RAM and CPU.
    INPUT
        df - encoded numerical dataframe to process
        n_rows - max number of rows for each set to process
    OUTPUT
        imputed_df - dataframe with imputed values
    '''
    
    # Split the dataframe in nr_set based on initial guess n_rows (close as possible to get equal-size)
    # This is necessary to keep the number of rows contained for the KNN imputer
    if n_rows < len(df):
        nr_set = len(df)/n_rows
        if len(df)%n_rows < n_rows/5:
            nr_set = np.floor(nr_set).astype('int32')
        else:
            nr_set = np.ceil(nr_set).astype('int32')
    else:
        nr_set = 1
    n_rows = np.ceil(len(df)/nr_set).astype('int32')
    
    imputer = KNNImputer()
    imputed_df= pd.DataFrame()
    for i in range(nr_set):
        start_idx = (n_rows)*i
        end_idx = n_rows*(i+1)
        imputed_df = imputed_df.append(pd.DataFrame(imputer.fit_transform(df[start_idx:end_idx])),
                          ignore_index = True)
    imputed_df.columns = df.columns
    return imputed_df


def prepare_data_w_KNN(df, resp_col):
    '''
    We now want to introduce multiple columns for each seaparate multiple choice.
    For these columns a missing answer results in 0 in all columns.
    We want to impute the missing values by means of kNN.
    In order to do that we first need to turn the categories into ordinals.
    After we impute the values, we want to return to the original categories
    in order to one-hot encode the dataframe for the modelling.
    We therefore want to split our dataframe in response y (Salary) and all the variables we selected (X). 
    
    INPUT
        df - dataframe containing the variables we want to use to make our predictions
        resp_col - string containing the column name contained in df that will be our response
    OUTPUT
        X - imputed and one-hot encoded dataframe
        y - response series
    '''
    X = encode_multiple_choices(df)
    X = X.reset_index().drop('index', axis=1)
    encoder = OrdinalEncoder()
    cat_cols = X.select_dtypes(include='object').columns

    revert_dict = {}
    for col in cat_cols:
        X[col], revert_dict[col] = encode_categories(X[col], encoder)
    print("Dataframe converted, ready to impute.")
    X = KNN_impute_dataframe(X, n_rows=20000)
    
    # we revert the categories to their original strings, but now we have the imputed values
    for col, dic in revert_dict.items():
        # before we revert we need to have only integers in the columns
        X[col] = np.around(X[col]).astype('int32')
        X[col].replace(dic, inplace=True)
    
    X = normalize(X)
    y = X[resp_col]
    X = X.drop([resp_col], axis=1)
    for col in cat_cols:
        X = pd.concat([X.drop(col, axis=1), pd.get_dummies(X[col], prefix=col, drop_first=False)], axis=1)
    print("Missing data imputed.")
    return X, y


def regression_model(X, y, model_type='best', random_state=10):
    '''
    Runs different regression model types and select the most appropriate.
    INPUT
        X - variables dataset
        y - response
        model_type (optional) - 'best' (default), 'linear', 'lasso', 'ridge', 'ElasticNet' or 'SGDRegressor'
        random_state (optional) - random state for pseudo randomic
    OUTPUT
        model - fitted linear model
        r2 - R-squared obtained with 30% of dataset as test dataset
        y_revert_dict - empty for numerical y, contains the revert dictionary for categorical y
    '''
    y_revert_dict = {}
    if pd.api.types.is_numeric_dtype(y) == False:
        encoder = OrdinalEncoder()
        y, y_revert_dict = encode_categories(y, encoder)
    models = ['linear', 'lasso', 'ridge', 'ElasticNet', 'SGDRegressor']
    if model_type not in models:
        model_result = 0
        model_type = 'linear'
        for model_sel in models:
            model = select_model(model_sel, random_state)
            model, r2, nr_test_samples = test_model(X, y, model, random_state)
            print("The r-squared score for the {} model was {} on {} values tested.".format(model_sel, r2, nr_test_samples))
            if r2 > model_result:
                model_type = model_sel
                model_result = r2
    model = select_model(model_type, random_state)
    model, r2, nr_test_samples = test_model(X, y, model, random_state)
    print("SELECTED: {} model -> r-squared score {} on {} values tested.".format(model_type, r2, nr_test_samples))
    return model, r2, y_revert_dict


def test_model(X, y, model, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=random_state)
    model.fit(X_train, y_train)
    y_test_preds = model.predict(X_test)
    r2 = r2_score(y_test, y_test_preds)
    return model, r2, len(y_test)


def select_model(model_type, random_state):
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'ElasticNet':
        model = ElasticNet()
    elif model_type == 'lasso':
        model = Lasso()
    elif model_type == 'ridge':
        model = Ridge()
    elif model_type == 'SGDRegressor':
        model = SGDRegressor(random_state=random_state, penalty='elasticnet')
    return model
    

def normalize(df):
    '''
    Used to normalize btw. 0 and 1 all numerical columns
    INPUT
        df - Dataframe to normalize
    OUTPUT
        result - Normalized dataframe
    '''
    result = df.copy()
    for feature_name in df.select_dtypes(include='number').columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if max_value != min_value:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        else:
            result[feature_name] = 1
    return result


def model_response(df, response, method=1):
    '''
    INPUT
        df - dataframe to model
        response - string, response column
        method (optional) - 1 (drop any row w/ missing values), 2 (mean value imputation) or 3 (kNN imputation). Default 1.
    OUTPUT
        X - dataframe with variables
        y - response series
        model - fitted model
        r2 - r-squared score
        coeffs_df - dataframe containing relative and absolute coefficients of linear model
    '''
    X = df.copy()
    if method == 1:
        # Method 1 - Drop any missing value row
        X = X.dropna()
        X = encode_multiple_choices(X)
        y = X[response]
        X.drop(response, axis=1, inplace=True)
        for col in X.select_dtypes(include='object').columns:
            X = pd.concat([X.drop(col, axis=1), pd.get_dummies(
                X[col], prefix=col, drop_first=False)], axis=1)
        X = normalize(X)
    elif method == 2:
        # Method 2 - Mean imputation
        X = encode_multiple_choices(X)

        for col in X.select_dtypes(include='object').columns:
            X = pd.concat([X.drop(col, axis=1), pd.get_dummies(
                X[col], prefix=col, drop_first=True)], axis=1)
        X.fillna(X.mean().to_dict(), inplace=True)
        y = X[response]
        X.drop(response, axis=1, inplace=True)
        X = normalize(X)
    elif method == 3:
        # Method 3 - Impute with k-Nearest Neighbors
        '''We also want to impute the missing values by means of kNN. 
        In order to do that we first need to turn the categories into ordinals.
        After we impute the values, we want to return to the original categories in order to one-hot encode 
        the dataframe for the modelling.'''
        X, y = prepare_data_w_KNN(X, response)

    model, r2, y_revert_dict = regression_model(X, y)
    coeffs_df = pd.DataFrame()
    coeffs_df['Parameter'] = X.columns
    coeffs_df['Weights'] = (model.coef_-np.mean(model.coef_)) / (np.max(model.coef_)-np.min(model.coef_))
    coeffs_df['Abs. Weights'] = np.abs(coeffs_df['Weights'])
    coeffs_df = coeffs_df.set_index('Parameter').sort_values('Abs. Weights', ascending=False)

    return X, y, model, r2, coeffs_df


def sample_countries(df, countries, nr_samples=50):
    '''
    Gets a dataframe with the same number of samples for each country.
    INPUT
        df - dataframe to process
        countries - series containing the Country information for the indexes in df
        nr_samples (optional) - int, number of samples by country to be subsetted
    OUTPUT
        sampled_df - subset dataframe
    '''
    # randomize
    process_df = df.sample(frac=1)
    if 'Country' not in process_df.columns:        
        process_df = process_df.merge(countries, left_index=True, right_index=True)
        print('Countries inserted.')
        
    countries = countries.unique()
    process_df = process_df[process_df['Country'].isin(countries)]
    
    # get the nr. of occurrencies from each country, if lower than nr_samples, set the nr_samples to the min
    min_counts = process_df['Country'].value_counts().min()
    if min_counts < nr_samples:
        nr_samples = min_counts
        print('Not enough samples. Selected: '+str(nr_samples))
    
    sampled_df = pd.DataFrame()
    for country in countries:
        subset = process_df[process_df['Country'] == country][:nr_samples]
        sampled_df = sampled_df.append(subset)
        
    sampled_df.drop('Country', axis=1, inplace=True)
    
    return sampled_df


def get_description(questions, year, column_name):
    '''
    Arguments:
        questions - dictionary containing the questions, 1-level keys are years, 2-level keys are df columns
        year - int of the year to be accessed in the dictionary
        column_name - string containing the parameter to search
    '''
    try:
        year = str(year)
        desc = questions[year][column_name]
    except:
        desc = 'Error in loading the text or data missing.'
    return desc


def search_kw(keyword, df_dict):
    '''
    Search among column header for the input keyword
    
    Argument:
        keyword = string to search in column headers
        df_dict = dictionary where keys are years and values are dataframe
    '''
    result = {}
    for key in df_dict.keys():
        result[key] = [col for col in df_dict[key].columns if keyword.lower() in col.lower()]
    return result


def add_value_labels(ax, spacing=0, fontsize=0):
    """
    Add labels to the end of each bar in a bar chart.
    Credit to https://stackoverflow.com/users/2161004/justfortherec
    
    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int) (optional): The distance between the labels and the bars.
        fontsize (int) (optional): The size of the font for the labels
    """

    # For each bar: Place a label
    nr_bars=len(ax.patches)
    if nr_bars <= 15:
        if fontsize == 0:
            fontsize = np.floor(-.3*nr_bars+14)
        if spacing == 0:
            spacing = np.floor(nr_bars-20)
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            # Number of points between bar and label. Change to your liking.
            space = spacing
            # Vertical alignment for positive values
            va = 'bottom'

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label and format number with one decimal place
            label = "{:.0f}".format(y_value)
            style = dict(fontsize=fontsize, color='white')

            # Create annotation
            ax.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(0, space),          # Vertically shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                ha='center',                # Horizontally center label
                va=va,                      # Vertically align label differently for positive and negative values.
                **style)


def analyze_column(df, questions, year, var):
    '''
    Return an analysis of the requested variable. Plot max 50 elements.
    
    Arguments:
        df: dictionary containing the dataframes containing the answers, keys are years
        questions: dictionary containing the questions, 1-level keys are years, 2-level keys are df columns
        year: integer of the year to access (key)
        var: column to be inspected in dataframes
    '''
    try:
        year = str(year)
        tot = df[year][var].shape[0]
        print('\''+var+'\'')
        print('Question: \''+get_description(questions, year, var)+'\'\n')
        print('Total elements: ',tot)
        numeric_cols = df[year].select_dtypes(include='number')
        if var not in numeric_cols:
            dfn = df[year][var].value_counts()
            print('Total valid elements: ',np.sum(dfn),'\n')
            print(dfn)
            ax = (dfn/tot*100).plot(kind='bar', title=var, figsize=(15,7));
            # variant
            # ax = (dfn/tot*100).plot(kind='bar', title=var, color=np.random.rand(df.shape[0],3));
            ax.set_ylabel('Percentage [%]');
            add_value_labels(ax)
        elif var in numeric_cols:
            # histogram is a better view
            try:
                dfn = df[year][var].dropna(axis=0)
                print('Total valid elements: ',dfn.shape[0], '\n')
                ax = dfn.hist(alpha=0.8, figsize=(10, 5), color='blue', bins=20, \
                    weights=np.ones_like(dfn) * 100. / tot);
                ax.set_ylabel('Percentage [%]');ax.set_title(var);
                #plt.tight_layout()
            except:
                print("Error during histogram creation.")
    except:
        print("Error: Data is missing.")

        
def rename_variable(df, questions, vars_to_convert, new_name):
    '''
    Rename the variable in the dataframe. Rename the variable in the first column of the schema.
    If the new column name is already present, it will be replaced.
    
    Arguments:
        df: dictionary containing the answers dataframes, keys are years
        questions: dictionary containing the questions, 1-level keys are years, 2-level keys are df columns
        vars_to_convert: dictionary in the form {year: var_to_be_replaced}
        new_name: string containing the new name of the variable
    '''
    for year in vars_to_convert.keys():
        
        if (vars_to_convert[year] != new_name):
            if year in df.keys():
                cols = df[year].columns
                if (vars_to_convert[year] in cols):
                    df[year] = df[year].drop([new_name], axis=1, errors='ignore')
            try:
                df[year] = df[year].rename({vars_to_convert[year]:new_name}, axis='columns')
                if year in questions.keys():
                    if vars_to_convert[year] in questions[year].keys():
                        questions[year][new_name] = questions[year].pop(vars_to_convert[year], None)
            except:
                continue
    return df, questions


def axis_not_scientific(plot,element):
    s = plot.state
    try:
        yaxis = s.select(dict(type=Axis, layout="left"))[0]
        yaxis.formatter.use_scientific = False
    except:
        pass
    try:
        xaxis = s.select(dict(type=Axis, layout="below"))[0]
        xaxis.formatter.use_scientific = False
    except:
        pass


def set_bokeh_plot(plot,element):
    '''
    Fix the axis for the Bokeh tools (PanTool, WheelZoomTool) and remove scientific notation
    '''
    # The bokeh/matplotlib figure
    s = plot.state
    pan_tool = s.select(dict(type=PanTool))
    pan_tool.dimensions="width"
    zoom_tool = s.select(dict(type=WheelZoomTool))
    zoom_tool.dimensions="width"
    axis_not_scientific(plot,element)
    
    # A dictionary of handles on plot subobjects, e.g. in matplotlib
    # artist, axis, legend and in bokeh x_range, y_range, glyph, cds etc.
    h = plot.handles
    
def plot_countries(df, col, round_val=1, col_tooltip='', nr_countries=10, reverse=False):
    '''
    Plot the overview for the top x (default 10) countries and for the overall countries as well.
    Returns a HoloViews plot layout.
        Arguments:
        df - Dataframe to process, must have the columns 'Country' and 'Year' within.
        col - Column in Dataframe where values are evaluated for the plotting process
        round_val (optional) - single numeric value to set the y axis limit on max found within col
        col_tooltip (optional) - tooltip to be set in the plots for the col values
        nr_countries (int) (optional) - number of countries to plot in the top views
        reverse (bool) (optional) - if True the bottom countries are listed 
    '''
    max_y = np.ceil(np.nanmax(df[col].values))
    max_y = max_y - max_y%round_val + 2*round_val
    if col_tooltip == '':
        col_tooltip = '@{'+col.replace(" ", "_")+'}{0,0.000}'    # Holoviews auto-replaces spaces with underscores
    years_list = list(df['Year'].unique())
    if reverse == True:
        label = 'Bottom'+str(nr_countries)
        plot_df = df[-nr_countries*len(years_list):]
    else:
        label = 'Top'+str(nr_countries)
        plot_df = df[:nr_countries*len(years_list)][::-1]
    plot_df_invert = plot_df[::-1].copy()
    df_invert = df[::-1].copy()
        
    # Plot settings and parameters
    top_hover = HoverTool(tooltips=[('Country', '@Country'),('Year','@Year'),(col,col_tooltip)])
    country_hover = HoverTool(tooltips=[("Year","@Year"),(col,col_tooltip)])
    year_hover = HoverTool(tooltips=[("Country","@Country"),(col,col_tooltip)])

    top_plot_arguments = dict(x='Year', y=col, by='Country', tools=[top_hover])
    options_shared = dict(height=700, ylim=(0, max_y), hooks=[set_bokeh_plot], active_tools=['wheel_zoom'], padding=(0.1,0.1))
    options = [opts.Bars(width=700, show_grid=True, **options_shared),
               opts.Scatter(xticks=years_list, marker = 'o', size = 10, **options_shared),
               opts.NdOverlay(width=650, xticks=years_list, **options_shared),
               opts.Layout(tabs=True)]

    # Create the multiplot
    layout = (plot_df_invert.hvplot(kind='barh', label=label+'BarPlot', **top_plot_arguments)+
              plot_df.hvplot(kind='line', label=label+'LinePlot', **top_plot_arguments)*
              plot_df.hvplot(kind='scatter', label=label+'LinePlot', **top_plot_arguments)+
              df.hvplot(kind='bar', x='Year', y=col, groupby='Country', label='SingleCountryDropdown',
                               tools=[country_hover])+
              df_invert.hvplot(kind='barh', x='Country', y=col, groupby='Year', label='AllCountriesYearSlider',
                               tools=[year_hover])
             ).opts(options)
    return layout


def plot_scatter(df, x, y, x_round_val=1, y_round_val=1, x_tooltip='', y_tooltip=''):
    '''
    
    Returns a HoloViews plot layout.
        Arguments:
        df - Dataframe to process, must have the column 'Country' adn the columns x and y within.
        x - Column in Dataframe where values are evaluated for the x-axis
        y - Column in Dataframe where values are evaluated for the y-axis
        x_round_val (optional) - single numeric value to set the x axis limits on max found within x
        y_round_val (optional) - single numeric value to set the y axis limits on max found within y
        x_tooltip (optional) - tooltip to be set in the plots for the x values
        y_tooltip (optional) - tooltip to be set in the plots for the y values
    '''
    max_y = np.ceil(np.nanmax(df[y].values))
    max_y = max_y - max_y%y_round_val + 2*y_round_val
    max_x = np.ceil(np.nanmax(df[x].values))
    max_x = max_x - max_x%x_round_val + 2*x_round_val
    '''
    if max_x > max_y:
        max_y = max_x
    else:
        max_x=max_y
    '''    
    if x_tooltip == '':
        x_tooltip = '@{'+x+'}{0,0.0}'
    if y_tooltip == '':
        y_tooltip = '@{'+y+'}{0,0.0}'

    # Plot settings and parameters
    hover = HoverTool(tooltips=[('Country', '@Country'),(x,x_tooltip),(y,y_tooltip)])
    padding = dict(x=(-1.2, 1.2), y=(-1.2, 1.2))
    
    options_shared = dict(width=700, height=700, xlim=(0, max_x), ylim=(0, max_y), hooks=[axis_not_scientific], 
                          active_tools=['wheel_zoom'], padding=(0.1,0.1), show_grid=True, show_legend=True,
                          legend_position='bottom', legend_cols=3)
    options = [opts.Scatter(marker = 'o', size = 10, fill_alpha=0.6, tools=[hover], color=hv.Palette('Set2'),
                            **options_shared),
               opts.Points(color='Country', cmap=cc.cm.fire, size=8, tools=[hover], **options_shared),
               opts.Labels(text_font_size='8pt', yoffset=y_round_val/5),
               opts.Overlay(**options_shared)
               ]
    
    ds = hv.Table(df)
    # Create the plot
    layout = (#hv.Scatter(df, kdims=[x], vdims=[y, 'Country'])*
              ds.to(hv.Scatter, x, y, 'Country').overlay()*
              hv.Labels(ds, kdims=[x, y], vdims=['Country'])
             ).opts(options)
    return layout


def plot_index_by_starting_string(df, start):
    '''
    INPUT
        df - dataframe, indexes to be searched
        start - string, case-sensitive search keyword
    '''
    match = pd.Series(df.index).str.startswith(start)
    if match.any():
        plot_arr = df.loc[df.index[match],:]
        plot_arr.plot.barh(figsize=(15,12));
        plt.gca().invert_yaxis();
        plt.grid('on');
    else:
        print('Not found.')

    
def fill_missing_incomes(df):
    '''
    Function to fill by linear extrapolation the missing years Avg. Income.
    Returns a new DataFrame.
    Arguments:
        df - DataFrame with columns 'Country', 'Avg. Income', 'Year'
    '''
    countries = list(df['Country'].unique())
    for c in countries:
        ds = df[df['Country']==c]
        ds_interp = ds.dropna(subset=['Avg. Income'])
        if len(ds_interp) > 1:
            interp_f = interpolate.interp1d(ds_interp['Year'].values,
                                            ds_interp['Avg. Income'].values,
                                            kind='linear', 
                                            fill_value="extrapolate")
            y = interp_f(2019)
            df.at[ds[ds['Year'] == 2019].index, 'Avg. Income'] = y
    return df


def sort_by_vals_in_col(df, col, method='last'):
    '''
    Sort the Countries in column 'Country' by the maximum values found in col and relative to column 'Year' for each of them.
    Returns a new sorted dataframe and a series of sorted countries.
    Arguments:
        df - Dataframe to process, must have the columns 'Country' and 'Year' within.
        col - Column in Dataframe where values are evaluated for the sorting process
        method (optional) - sorting method to apply over the year to get the ranking 'max', 'mean', 'last'. Defaults to last valid.
    '''
    if method == 'mean':
        sorted_countries = df.groupby('Country')[col].mean().sort_values(ascending=False)
    elif method == 'max':
        sorted_countries = df.groupby('Country')[col].max().sort_values(ascending=False)
    else:
        sorted_countries = df.groupby('Country')[col].last().sort_values(ascending=False)
    sorted_df = df.copy()
    sorted_df['Country'] = df['Country'].astype("category").cat.set_categories(sorted_countries.index.to_list())
    sorted_df = sorted_df.sort_values(['Country','Year'], ignore_index=True)
    sorted_df['Country'] = sorted_df['Country'].astype("object")
    return sorted_df, sorted_countries

