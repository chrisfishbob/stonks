import pandas as pd

if __name__ == "__main__":
    vti = pd.read_csv("./data/NB/VTI.csv")
    vti = vti[["Date", "Change"]]
    vti["Date"] = pd.to_datetime(vti["Date"]).dt.strftime("%Y-%m-%d")
    vti["Market_Change"] = vti["Change"]
    vti = vti[["Date", "Market_Change"]]

    moon = pd.read_csv("./data/NB/moon_illumination.csv")
    moon["Date"] = pd.to_datetime(moon["date"]).dt.strftime("%Y-%m-%d")
    moon = moon[["Date", "phase"]]

    temp = pd.read_csv("./data/NB/nyc_temp.csv")
    temp = temp[["Date", "TAVG", "PRCP"]]
    temp["Date"] = pd.to_datetime(temp["Date"]).dt.strftime("%Y-%m-%d")
    temp["Temp"] = pd.cut(
        temp["TAVG"],
        bins=[-float("inf"), 45.0, 75.0, float("inf")],
        labels=["Cold", "Mild", "Hot"],
        right=False,
    )
    temp["Weather"] = pd.cut(
        temp["PRCP"],
        bins=[-float("inf"), 0.1, 0.5, float("inf")],
        labels=["Clear", "Misty", "Rainy"],
        right=False,
    )
    temp = temp[["Date", "Temp", "Weather"]]

    tomato = pd.read_csv("./data/NB/Tomato.csv")
    tomato = tomato[["Date", "Average"]]
    tomato["Date"] = pd.to_datetime(tomato["Date"]).dt.strftime("%Y-%m-%d")
    tomato["Daily_Change"] = tomato["Average"].diff()
    tomato["Tomato_Change"] = tomato["Daily_Change"].apply(
        lambda x: "Up" if x > 0 else ("Down" if x < 0 else "No Change")
    )
    tomato = tomato[["Date", "Tomato_Change"]]

    """stock_data = pd.read_csv('./data/stock_data.csv')
    stock_data['Date'] = pd.to_datetime(stock_data['timestamp'] / 1000, unit='s').dt.strftime('%Y-%m-%d')
    stock_data['Daily_Change'] = stock_data['close'].diff()
    stock_data['Stock_Change'] = stock_data['Daily_Change'].apply(lambda x: 'Up' if x > 0 else ('Down' if x < 0 else 'No Change'))
    stock_data = stock_data[['Date', 'ticker', 'close', 'Stock_Change']]
    print(stock_data)"""

    merged_df = pd.merge(vti, moon, on="Date", how="inner")
    merged_df = pd.merge(merged_df, temp, on="Date", how="inner")
    merged_df = pd.merge(merged_df, tomato, on="Date", how="inner")
    # merged_df = pd.merge(merged_df, stock_data, on='Date', how='inner')

    # most_frequent_combinations = merged_df.groupby(['phase', 'Temp', 'Weather', 'Tomato_Change', 'Market_Change'], observed=True)['Stock_Change'].apply(lambda x: x.mode()[0]).reset_index()

    # print(merged_df)
    # print(most_frequent_combinations)
    merged_df.to_excel("./data/NB/NB_data_2.xlsx", index=False)
    # print(most_frequent_combinations)
    # most_frequent_combinations.to_excel('./data/NB/NB_data_2.xlsx', index=False)
