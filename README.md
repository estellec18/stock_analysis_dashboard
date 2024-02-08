# stock_analysis_dashboard

*projet élaboré afin de se familiariser à la gestion des time series*

Objectif : création d'un dashboard d'analyse d'actions, sur la base des informations disponibles via l'API yahoo finance (données issues des marchés financiers quasi en direct).

3 modules ont été développés:
- stock_viz.py : permet de visualiser l'action analysée et de le comparer avec d'autres actifs
- stock_ml.py : s'appuie sur des algorithmes de machine learning pour
    - extraire un sentiment analysis des news concernant une action
    - prédire la direction des prix (up or down)
    - prédire le prix de fin de mois
    - prédire le prix sur une période donnée
- stock_portfolio.py : permet d'analyser les performances de portefeuilles d'action et de les optimiser

Ci-dessous, quelques captures d'écran préliminaires du dashboard

![Capture d’écran 2024-01-31 à 16 00 23](https://github.com/estellec18/stock_analysis_dashboard/assets/126951321/3d893a7c-66b1-4870-b994-6744d7d33e47)

#

![Capture d’écran 2024-01-31 à 16 01 12](https://github.com/estellec18/stock_analysis_dashboard/assets/126951321/317da6f6-e8d8-4188-851e-45263d4b7bdf)

#

![Capture d’écran 2024-01-31 à 16 01 29](https://github.com/estellec18/stock_analysis_dashboard/assets/126951321/73e229c0-7cac-48c4-8cf3-7b1e2cf94987)

#

![Capture d’écran 2024-01-31 à 16 02 26](https://github.com/estellec18/stock_analysis_dashboard/assets/126951321/67319681-935d-49a9-81f8-2a6f5e1450b7)

#

![Capture d’écran 2024-01-31 à 16 03 24](https://github.com/estellec18/stock_analysis_dashboard/assets/126951321/c356cb82-11a3-4df7-adb1-61c922ba5a50)

#

![Capture d’écran 2024-01-31 à 16 04 24](https://github.com/estellec18/stock_analysis_dashboard/assets/126951321/0546113b-7e4a-4dd3-b4ce-a23a6f55410f)


**LIMITES**

- Il est très compliqué, voir peu pertinent, de prédire le prix d'actions à partir d'algorithmes de machine learning.
Cela s'explique par la nature même de la data et ce qui drive sa fluctuation : de nombreux facteurs externes interviennent (évènement macroéconomique, comportement irrationnel,...) et se combinent pour rendre les prix des actions dynamiques et volatiles, avec des patterns peu répétitifs.

- Il serait interessant d'introduire des éléments macroéconomiques (inflation rate, unemployment rate...) comme feature du modele en plus du prix de l'action lui même. Cela est impossible dans la mesure où les KPI macro sont calculés pour la période n-1 et ne seraient donc pas disponibles au moment de l'inference (hindsight bias).
