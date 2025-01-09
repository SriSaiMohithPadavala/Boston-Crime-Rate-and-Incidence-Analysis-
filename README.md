# Boston-Crime-Rate-and-Incidence-Analysis

**Project Overview**

Public safety is a critical concern in Boston, particularly in districts with diverse cultural and economic profiles. This project investigates the correlation between police distribution and crime rates across Boston, emphasizing districts with high population density, poverty, and income disparities. The aim is to identify patterns, test hypotheses, and develop actionable recommendations to enhance public safety and resource allocation.

**Importance of Study**

Boston's position as a prominent urban hub necessitates effective crime prevention strategies. By analyzing socio-economic data and crime incidents, this study seeks to provide a framework for equitable policing and improved safety in high-risk areas.

**Motivation**

Understanding the underlying socio-economic drivers of crime allows for the formulation of targeted interventions. The ultimate goal is to mitigate crime, enhance resource utilization, and improve community living standards.

## **Approach**

**Data Collection**

- Primary Data Set: 
  - [Boston Crime Incident Reports](https://data.boston.gov/dataset/crime-incident-reports-august-2015-to-date-source-new-system): (August 2015- 2022)
  
  Features: Crime type, location, and time.
- Auxilliary Data Set:
  - [Boston Police Department Resource Allocation](https://www.policedatainitiative.org/participating-agencies/boston-massachusetts-police/)
  - [Neighborhood Demographics](https://data.boston.gov/dataset/neighborhood-demographics)

**Data Preparation**
- Removal of null and duplicate values
- Standardization of crime categories into 29 groups
- Exclusion of non-criminal records
- Categorization of crimes by severity levels (1–4)
- Augmentation with socio-economic data using demographic indicators

### Data Analysis and Methodologies

1. Exploratory Data Analysis (EDA): 
    - Bar plots for district-wise crime distribution
    - Seasonal and monthly trends
    - Time-series analysis using ARIMA and Prophet models
2. Hypothesis Testing:
    - ANOVA for seasonal variations in crime
    - T-tests for income level impacts on crime severity
3. Predictive Modeling:
    - RandomForest for crime severity prediction
    - Evaluation with confusion matrices and classification reports
4. Spatial and Demographic Analysis:
    - Heatmaps for crime intensity visualization
    - Scatter plots to correlate income and crime rates
    - Word clouds for high-probability crimes
  

 #### **Tools and Techniques Used**

- Data Processing: Python, Excel
- Visualization: Matplotlib, Seabon, Tableau
- Statistical Modeling: Scipy, Statsodels
- Web Scrapning: Python libraries for data extraction


##### **Outcomes**

1. Identified high-crime districts: Roxbury, Dorchester, Mattapan.
2. Highlighted key issues such as socio-economic disparities, resource allocation gaps, and crime trends.
3. Recommendations for:
    - Strategic police staffing and resource redistribution
    - Community engagement and youth programs
    - Economic development initiatives
  
##### Visualizations
1. Crime Trends Over Time: Bar and line graphs depicting yearly and monthly patterns.
2. Heatmaps: Highlighting high-crime zones.
3. Demographic Analysis: Radar charts comparing age, income, and ethnicity distribution across districts.
4. Predictive Insights: Time-series forecasts for future crime incidents.



      
  


 
