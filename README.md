


# 🧠 Smart Inventory Forecasting  
*A practical data forecasting project combining real-world datasets, predictive modeling, and visualization.*

---

## 🪄 Overview  
This project is all about **predicting product demand** using time-series forecasting.  
I worked on two related datasets to explore how forecasting can help businesses maintain better inventory levels and prevent overstocking or shortages.

It’s divided into two parts:
1. **Automobile Demand Forecasting** — using synthetic automobile sales data (custom-built dataset).  
2. **Store Item Demand Forecasting (Kaggle Dataset)** — applying Prophet to real-world retail sales data to forecast item-wise demand across stores.

Both analyses are based on **Facebook Prophet**, which models trends, weekly and yearly seasonality, and produces confidence intervals for predictions.

Through these two studies, my goal was to understand:
- How demand fluctuates over time,
- How to visualize patterns like seasonality and trends,
- And how predictive models can assist businesses in restocking decisions.

---

## 🏎️ Part 1: Automobile Demand Forecasting  

### 🎯 Objective  
Forecast the sales demand for various automobile spare parts such as brake pads, oil filters, air filters, spark plugs, etc.  
The idea was to help a retailer predict **which automobile components are likely to go out of stock soon**, based on past demand.

### 🧩 Dataset Details  
- **Type:** Custom synthetic dataset created for experimentation.  
- **Columns:**  
  - `Date` — Daily sales date  
  - `Product` — Item name (e.g., Brake Pad, Spark Plug)  
  - `Units_Sold` — Number of units sold per day  

### ⚙️ Approach  
- Aggregated daily sales data for each product.  
- Trained a **Prophet model** per product to capture sales trend and seasonal cycles.  
- Forecasted demand for the next 30 days.  
- Calculated % change between recent and forecasted periods to identify stock risks.

### 📈 Key Insights  
- **Brake Pads and Car Batteries** showed strong seasonality patterns, likely linked to service cycles.  
- The forecast visualization helped identify which products might face a **demand spike** or **drop** soon.  
- Top 5 items with highest predicted demand were visualized using bar charts.

### 🖼️ Sample Output  
![Automobile Forecast Chart](https://user-images.githubusercontent.com/example/automobile_forecast.png)  
*(Sample visualization of daily predicted demand for Brake Pads)*

---

## 🏬 Part 2: Store Item Demand Forecasting (Kaggle Dataset)  

### 🎯 Objective  
Use Kaggle’s **Store Item Demand Forecasting Challenge** dataset to forecast the next 90 days of sales for each store-item combination.  
This dataset provided a more realistic scenario of retail demand with large-scale daily sales.

### 🧩 Dataset Details  
- **Source:** [Kaggle – Store Item Demand Forecasting Challenge](https://www.kaggle.com/c/demand-forecasting-kernels-only/data)  
- **Files Used:**  
  - `train.csv` — historical sales data  
  - `test.csv` — dates to forecast  
  - `sample_submission.csv` — submission format for forecasts  
- **Columns:**  
  - `date` — Sales date  
  - `store` — Store ID  
  - `item` — Item ID  
  - `sales` — Units sold  

### ⚙️ Approach  
- Loaded and cleaned the dataset with **Pandas**.  
- Created a Prophet model for each store-item combination.  
- Modeled **trend**, **weekly**, and **yearly seasonality** patterns.  
- Forecasted demand for the next 90 days.  
- Identified **top 5 items per store** with highest future average daily sales.

### 🧠 Example Insight  
For **Store 1**, the model predicted that:
- Item `2` and Item `3` have the highest future demand.
- Demand patterns show a consistent upward trend from 2013–2018.
- Mondays typically have the lowest sales, while weekends peak slightly higher.

### 🖼️ Forecast Components Example  
| Trend | Weekly Pattern | Yearly Seasonality |
|-------|----------------|-------------------|
| ![trend](https://user-images.githubusercontent.com/example/trend.png) | ![weekly](https://user-images.githubusercontent.com/example/weekly.png) | ![yearly](https://user-images.githubusercontent.com/example/yearly.png) |

### 🧾 Forecast Result  
![Top5Items](https://user-images.githubusercontent.com/example/top5items.png)  
*Top 5 forecasted items by demand for Store 1.*

---

## 🛠️ Tools & Libraries Used  
| Category | Libraries / Tools |
|-----------|-------------------|
| Forecasting | **Facebook Prophet** |
| Data Analysis | **Pandas**, **NumPy** |
| Visualization | **Matplotlib**, **Seaborn** |
| Development | **Jupyter Notebook**, **GitHub** |
| Dataset Source | **Kaggle**, custom CSVs |

---

## 🧮 Folder Structure  

smart-inventory-forecasting/
│
├── notebooks/
│   ├── 01_Automobile_Demand_Forecasting.ipynb
│   └── 02_Store_Item_Demand_Forecasting.ipynb
│
├── data/
│   ├── sales_data.csv
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── outputs/
│   ├── product_demand_summary.csv
│   ├── Brake Pads_forecast.csv
│   ├── store1_top5_forecast_summary.csv
│
└── README.md

---

## 💡 Key Learnings  
- Understood how **Prophet handles trend, seasonality, and uncertainty intervals**.  
- Learned to handle **real-world time-series data** efficiently.  
- Experienced the importance of **data visualization** in interpreting forecast results.  
- Improved my workflow with **GitHub and Jupyter notebooks** for clean project documentation.  

---

## 🚀 Future Improvements  
- Integrate **ARIMA / LSTM** models to compare Prophet’s performance.  
- Create a **Streamlit dashboard** for real-time forecasting visualization.  
- Expand datasets and automate retraining when new sales data arrives.  

---

## 🙋‍♂️ About Me  
I’m **Madhan Kumar Tammineni**, a Master’s student in Computer Science at the **University of Memphis**.  
I’m passionate about **data science, AI, and building practical tools** that bridge analytical thinking with real-world impact.  
This project helped me deepen my understanding of forecasting and its potential role in decision-making systems.

📧 [tammineni.madhan.kumar@gmail.com](mailto:tammineni.madhan.kumar@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/madhan-kumar-tammineni-4487a4197)  
💻 [GitHub](https://github.com/Madhan120-prog)

---

### ⭐ If you found this project interesting, feel free to star the repo!  
Every bit of encouragement helps me continue learning and building better projects 🌱  
