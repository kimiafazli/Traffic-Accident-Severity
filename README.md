# Identifying Key Factors in Traffic Accident Severity: A SHAP and Machine Learning Approach in North Carolina
 Road accidents present a significant challenge globally, impacting economies, national resources, and human lives. This study focuses on analyzing traffic accident data from North Carolina to identify the factors influencing crash severity. Utilizing data collected over several years, five machine learning techniques including Logistic Regression, XGBoost, Random Forest, K-Nearest Neighbors, and Decision Tree were rigorously evaluated. Among these, XGBoost demonstrated superior performance, achieving an accuracy of 91%. Detailed analyses, including Confusion Matrix Analysis and ROC-AUC Analysis, further validated the effectiveness of the algorithms. Additionally, SHAP values analysis was employed to pinpoint key factors contributing to accident severity. These findings highlight the critical role of diverse features in accurately predicting road accident severity and emphasize the potential of advanced machine learning methods in enhancing road safety measures.
* Dataset
 [kaggle datasets download -d sobhanmoosavi/us-accidents](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)


* Results
  
![Screenshot (1187)](https://github.com/user-attachments/assets/39cb6ff2-e1b3-456a-8e9c-40e2dea00720)

![image](https://github.com/user-attachments/assets/bbeb8187-a7d1-47ac-afd7-a968b1a65bb3)

![image](https://github.com/user-attachments/assets/3745a643-a4f2-474d-8c11-265bdfd2901d)

![image](https://github.com/user-attachments/assets/986a9932-eddb-4594-9ee6-6fcfd5b327ef)

* Insights
  
![image](https://github.com/user-attachments/assets/41066a98-87a5-4dc0-a0a8-29144809db14)

![image](https://github.com/user-attachments/assets/83f027e2-2f18-473c-8379-5eab662578f6)

![image](https://github.com/user-attachments/assets/ddbc38ea-88c1-412f-8c34-40affa2f5a5f)

![Screenshot (1186)](https://github.com/user-attachments/assets/6634511e-0af4-4b94-9b02-8144001ce1db)



* DISCUSSION, IMPLICATIONS, AND FUTURE RESEARCH
  In this study, we explored road safety in North Carolina using machine learning to predict accident severity and identify key contributing factors. Our SHAP analysis revealed that the most significant factors influencing accident severity include 'Distance (mi)', the year and month of the incident, accident duration, and geospatial features such as latitude and longitude. The length of the road affected by the accident emerged as the most critical factor, with longer stretches correlating with increased severity. The timing of the incident, both in terms of the year and month, also significantly impacts severity, highlighting evolving trends and seasonal patterns. Longer accident durations are linked to higher severity, underscoring the importance of prompt emergency responses. Additionally, geospatial features play a crucial role in determining accident severity. Other influential factors include socioeconomic and demographic elements, traffic signals, and weather conditions, though they have a lesser impact. To address these issues, we recommend focusing on road maintenance and safety enhancements, especially on road sections that are frequently involved in accidents. Implementing warning signs and speed control measures in these areas can help reduce accident frequency and severity. Regular updates to traffic safety measures based on historical data are essential to proactively address future hotspots. Seasonal safety campaigns can raise public awareness about specific driving hazards during different times of the year, such as icy roads in winter or increased traffic during holidays. Improving emergency response times through better coordination and real-time monitoring systems can significantly reduce the duration and impact of accidents. Geospatial analysis can identify high-risk areas, guiding localized safety measures such as better lighting, improved signage, and road design adjustments. Developing a comprehensive traffic management system that integrates historical and real-time data will enable dynamic and adaptive traffic control. Launching targeted education campaigns, advocating for policy changes, and investing in infrastructure improvements will further enhance road safety. Additionally, promoting the adoption of advanced vehicle technologies can assist drivers in real-time hazard detection and avoidance. For future research, it is recommended to explore additional machine learning models and incorporate more granular data, such as specific driver behavior and vehicle condition, to further enhance prediction accuracy. Investigating the long-term effects of implemented safety measures and continuous monitoring can provide deeper insights into their effectiveness and guide ongoing improvements in road safety strategies.
