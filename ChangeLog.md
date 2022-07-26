-----------------------------------------------------------------------------------------------------------------------------------------------------------------

Date:7/18/2022
--
Updates:

--runCPCA.py-- 
the basic CPCA code with input Z, G or H matrix(in '.mat' file) to return an output python dictionary with the basic CPCA calculation results.
include two functions: runCPCA and findLoadingsScores

--G_CPCA_Analysis.py--
the major work of this code is to calculate the predictor loadings and add relevant results to the python dictionary.
include funtion: G_CPCA_Analysis

--loading plot.ipynb--
to generate heatmap for loading matrix(components or predictors)

--Scree_plot.ipynb--
to generate scree plot for all components

-----------------------------------------------------------------------------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------------------------------------------------------------------------------

Date:7/26/2022
--
Updates:

--splitHalfCrossValidCpca.py--
Preparation for Cross Validation, split the raw data into training set and testing set.

--getPcaPredCorr.py--
Run CPCA on the data, input the number of components according to the scree plot and calculate the predictor scores and loadings.

--SelectCompLoadingsBig.py--
Test the reliability of all variables in Z seperately.

--CPCA_CV_main.ipynb--
Do Cross Validation CPCA on the data and generate the report.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------
