# World Bank’s Climate Change Data Analysis

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Loading the data
# creating a function to load and clean the data
def clean_data(csv_path):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, skiprows=4)
    else:
        raise FileNotFoundError("File path doesn't exits.")

    # droping the last unnamed column
    df = df.drop(df.columns[-1], axis=1)

    # printing first five rows
    print(df.head())

    # converting all the year columns to a single column "Year"
    df = pd.melt(df, id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                        var_name='Year', value_name='Value')

    # convertign the year to int
    df['Year'] = df['Year'].astype("int")

    '''
    I will select following **indicators** for the analysis :
    `Population, total`, `Urban population (% of total population)`,
    `Total greenhouse gas emissions (% change from 1990)`,
    `Total greenhouse gas emissions (kt of CO2 equivalent)`,
    `Energy use (kg of oil equivalent per capita)`,
    `Electric power consumption`, 
    `Renewable energy consumption (% of total final energy consumption)`, 
    `Access to electricity (% of population)`,
    `Foreign direct investment, net inflows (% of GDP)`, 
    `Agricultural land (% of land area)`, 
    `Forest area (% of land area)`

    And I will analyse following **contries** : 
    United States, Australia, India, China, 
    Russian Federation, Germany, South Africa, 
    United Kingdom, Brazil, France, Japan, Switzerland
    '''

    # creating a country and indicator list to select them
    country_list = ["United States", "Australia", "India", "China",
                "Russian Federation", "Germany", "South Africa",
                "United Kingdom", "Brazil", "France", "Japan", "Switzerland"]

    indicator_list = ["Population, total",
                    "Urban population (% of total population)",
                    "Total greenhouse gas emissions (% change from 1990)",
                    "Total greenhouse gas emissions (kt of CO2 equivalent)",
                    "Energy use (kg of oil equivalent per capita)",
                    "Electric power consumption (kWh per capita)",
                    "Renewable energy consumption (% of total final energy consumption)",
                    "Foreign direct investment, net inflows (% of GDP)",
                    "Agricultural land (% of land area)",
                    "Arable land (% of land area)",
                    "Forest area (% of land area)"]

    # filtering the dataframe based on the selected countries and indicators
    df = df[df["Country Name"].isin(country_list)]
    df = df[df["Indicator Name"].isin(indicator_list)]

    # as Country Code and Indicator code as alias I will remove them
    df = df.drop(columns=['Country Code', 'Indicator Code'])

    # Filtering for the years
    df = df[df['Year']>=1990]

    # data info
    print(df.info())

    # missing values
    print(df.isna().sum())

    # creating the data fromes for years and countries
    # "df" is the dataframe for the years
    # creating 2nd dataframe for countries as column

    df_countries = df.pivot(index=['Year', 'Indicator Name'], columns='Country Name', values='Value')
    df_countries = df_countries.reset_index()

    return df, df_countries


# loading the csv file for World Bank's data
df, df_countries = clean_data("API_19_DS2_en_csv_v2_6183479.csv")

# Data Analysis

# some statistical analysis
# Pivot the DataFrame to have separate columns for Population and Emissions
# The data is available till 2020, so I will filter till this year
df_pivot = df.pivot_table(index=['Country Name', 'Year'], columns='Indicator Name', values='Value').reset_index()
df_pivot = df_pivot[df_pivot['Year']<=2020]


print(df_pivot.head())

# analysing the population
print(df_pivot.groupby("Year").sum(["Population, total"])["Population, total"].describe())

# 1. Population and Greenhouse emission

# The emission values are very huge and difficult to distinguish,
# so I will create a new column to compare with the maximum
# emission of that country


# comaparing with the maximum of the respective country
df_pivot['Greenhouse emission, wrt 1990'] = df_pivot.groupby('Country Name')['Total greenhouse gas emissions (kt of CO2 equivalent)'].transform(lambda x: x - x[df_pivot['Year'] == 1990].values[0])

# population wrt 1990 for easy understanding
df_pivot['Population, wrt 1990'] = df_pivot.groupby('Country Name')['Population, total'].transform(lambda x: x - x[df_pivot['Year'] == 1990].values[0])

# convertign poplulation to million
df_pivot['Population, wrt 1990'] = df_pivot['Population, wrt 1990']/1e6

# plotting
plt.figure()
sns.scatterplot(x='Year', y='Population, wrt 1990', data=df_pivot,
                size="Greenhouse emission, wrt 1990", hue="Country Name",
                markers=True, legend="auto", sizes=(0, 200), palette="tab20")

# setting the labels and title
plt.xlabel('Year')
plt.ylabel('Population, wrt 1990 (million)')
plt.title('Population and Greenhouse Gas Emissions Over Time')

# moving the legend outside the plot
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

plt.savefig("Population and Greenhouse emission.png", bbox_inches='tight')
plt.show()


# 2. Power usages and Green house gas

# plotting
plt.figure()
sns.lineplot(x='Year', y='Renewable energy consumption (% of total final energy consumption)',
             data=df_pivot, hue="Country Name", markers=True, palette="tab20")

# setting the labels and title
plt.xlabel('Year')
plt.ylabel('Renewable energy consumption \n (% of total final energy consumption)')
plt.title('Renewable Energy Consumption Over Time')

# moving the legend outside the plot
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

plt.savefig("Renewable Energy Consumption Over Time.png", bbox_inches='tight')
plt.show()


# 3. Urban population, Agricultural and Forest
# analysing India's urban population, agriculture and forest
df_India = df_pivot[df_pivot['Country Name']=="India"]

columns_to_keep = ["Year",
                   'Urban population (% of total population)', 
                   'Agricultural land (% of land area)',
                   'Forest area (% of land area)']

df_India_agri = df_India[columns_to_keep]
df_India_agri = df_India_agri.set_index("Year")

# understsnding the correlation
print("The correlation is : ")
print(df_India_agri.corr())

# calculating the percentage for each indicator for plotting
df_percentage = df_India_agri.div(df_India_agri.sum(axis=1), axis=0) * 100

# plotting
plt.figure()
# plotting the stacked bar chart
ax = df_percentage.plot(kind='bar', stacked=True, figsize=(6, 4), colormap='tab20')

# setting labels and title
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.title('Urbanisation effect on agriculture for India')

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.savefig("Urban Population, Agriculture and Forest Area.png",
bbox_inches='tight')
plt.show()


# 4. Areable land comparison

df_areable = df_pivot[["Country Name",
                   "Year",
                   "Arable land (% of land area)"]]

# analysing the areable land and greenhouse gas emission for IND and USA
df_areable = df_areable[df_areable["Country Name"].isin(['India', 'United States'])]

# plotting
plt.figure()
sns.lineplot(x="Year", y="Arable land (% of land area)", data=df_areable, hue="Country Name")

# setting labels and title
plt.xlabel('Year')
plt.ylabel('Arable land (% of land area)')
plt.title('Arable land over the years')

plt.savefig("Arable land over the years.png", bbox_inches='tight')
# Show the plot
plt.show()


# 5. Contribution to Greenhouse gas emission
# greenhouse gas emmited per capita for year 2020
selected_year = 2020

# filtering data for the selected year
df_selected_year = df_pivot[df_pivot['Year'] == selected_year]

# Calculate emissions per person
df_selected_year['Emissions per Person'] = df_selected_year['Total greenhouse gas emissions (kt of CO2 equivalent)'] / df_selected_year['Population, total']

colors = sns.color_palette('tab10', n_colors=len(df_selected_year))
# Create a pie chart
plt.figure(figsize=(10, 8))
plt.pie(df_selected_year['Emissions per Person'], labels=df_selected_year['Country Name'], autopct='%1.1f%%', startangle=90, colors=colors)
plt.title(f'Contribution to Total Greenhouse Gas Emissions per capita ({selected_year})')
plt.savefig("Greenhouse gas contribution2020.png", bbox_inches='tight')
# Show the pie chart
plt.show()

# greenhouse gas emmited per capita for year 1990
selected_year = 1990

# Filter data for the selected year
df_selected_year = df_pivot[df_pivot['Year'] == selected_year]

# Calculate emissions per person
df_selected_year['Emissions per Person'] = df_selected_year['Total greenhouse gas emissions (kt of CO2 equivalent)'] / df_selected_year['Population, total']

colors = sns.color_palette('tab10', n_colors=len(df_selected_year))
# Create a pie chart
plt.figure(figsize=(10, 8))
plt.pie(df_selected_year['Emissions per Person'], labels=df_selected_year['Country Name'], autopct='%1.1f%%', startangle=90, colors=colors)
plt.title(f'Contribution to Total Greenhouse Gas Emissions per capita ({selected_year})')
plt.savefig("Greenhouse gas contribution1990.png", bbox_inches='tight')
# Show the pie chart
plt.show()


# 6. Correlation of the Indicators
# I will only consider those columns whose correlation
# is not that direct

correlation_columns = ['Urban population (% of total population)',
                       'Agricultural land (% of land area)',
                       'Forest area (% of land area)',
                       'Electric power consumption (kWh per capita)',
                       'Foreign direct investment, net inflows (% of GDP)',
                       'Renewable energy consumption (% of total final energy consumption)',
                       'Total greenhouse gas emissions (% change from 1990)']

# alias names for the columns
alias_names = ["Urban population", "Agriculture land", "Forest area",
               "Electricity consumption", "FDI", "Renewable energy",
              "Greenhouse gas"]

# correlation for India
df_India_corr = df_pivot[df_pivot['Country Name']=="India"]
df_India_corr = df_India_corr[correlation_columns]
df_India_corr.columns = alias_names

# plotting
plt.figure()
sns.heatmap(df_India_corr.corr(), cmap="Blues", annot=True)
plt.title('Indicator correlation for India', fontsize=13)
plt.savefig("Indicator correlation for India.png", bbox_inches='tight')
plt.show()


# correlation for China
df_China_corr = df_pivot[df_pivot['Country Name']=="China"]
df_China_corr = df_China_corr[correlation_columns]
df_China_corr.columns = alias_names

# plotting
plt.figure()
sns.heatmap(df_China_corr.corr(), cmap="Blues", annot=True)
plt.title('Indicator correlation for China', fontsize=13)
plt.savefig("Indicator correlation for China.png", bbox_inches='tight')
plt.show()