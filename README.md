# Warehouse Activity Profiling Simulator

A comprehensive Streamlit application for warehouse data analysis, slotting optimization, and scenario simulation.

## Team Members

- **Sultan Albinali** - B00103378
- **Mohamad Ftouni** - B00089796
- **Fawaz Al Jandali** - B00082922
- **Ahmed Abusnina** - B00095469

## Course Information

**Course:** INE 494-5: Analysis of Procurement and Warehousing Operations  
**Institution:** American University of Sharjah - Industrial Engineering Department

## Features

- **ABC Classification:** Demand-based SKU analysis
- **Clustering Analysis:** K-means clustering of SKUs by demand, weight, and volume
- **Association Mining:** Discover frequently co-purchased items
- **Slotting Optimization:** Demand-based warehouse slotting with zone utilization
- **Scenario Simulation:** What-if analysis with adjustable demand and capacity multipliers
- **Interactive Visualizations:** Plotly charts and network graphs
- **Excel Export:** Download comprehensive reports

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the application locally:
```bash
streamlit run WarehousingChallenge.py
```

The app will open in your browser at `http://localhost:8501`

## Data Requirements

Upload an Excel file with the following sheets:
- **Orders:** Order information
- **Lines:** Order line items
- **SKU_Master:** SKU details (ID, Category, Demand, Weight, Volume, Storage Type)
- **Storage_Zones:** Zone information (ID, Type, Capacity, Distance)

Alternatively, place `Warehouse_Synthetic_Dataset_2025.xlsx` in your Downloads folder.

## License

Academic project for educational purposes.
