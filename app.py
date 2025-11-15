import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import io
import plotly.express as px

# -------------------------------------------------
# Helper Functions
# -------------------------------------------------
def expand_to_room_nights(df):
    rows = []
    for _, r in df.iterrows():
        if pd.isna(r["from_date"]) or pd.isna(r["to_date"]):
            continue
        start = pd.Timestamp(r["from_date"]).floor("D")
        end = pd.Timestamp(r["to_date"]).floor("D")
        if end <= start:
            continue
        for d in pd.date_range(start, end - pd.Timedelta(days=1)):
            rows.append({
                "date": d,
                "room_type": r.get("new_title", None),
                "ex_tax": r.get("ex_tax", 0),
                "tax_amount": r.get("tax_amount", 0),
                "revenue": r.get("total", 0),
                "nights": 1
            })
    return pd.DataFrame(rows)


def lstm_forecast(series, n_steps=7, epochs=50):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    X, y = [], []
    for i in range(n_steps, len(scaled)):
        X.append(scaled[i - n_steps:i])
        y.append(scaled[i])
    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(64, activation='relu', input_shape=(n_steps, 1)),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.fit(X, y, epochs=epochs, verbose=0)
    return model, scaler, n_steps


def forecast_future(model, scaler, series, n_steps, future_days=30):
    scaled_data = scaler.transform(series.values.reshape(-1, 1))
    input_seq = scaled_data[-n_steps:].reshape(1, n_steps, 1)
    preds = []
    for _ in range(future_days):
        pred = model.predict(input_seq, verbose=0)
        preds.append(pred[0][0])
        input_seq = np.append(input_seq[:, 1:, :], [[[pred[0][0]]]], axis=1)
    preds = np.array(preds).reshape(-1, 1)
    preds = scaler.inverse_transform(preds)
    return preds.flatten()






# -------------------------------------------------
# Streamlit UI (with SIDEBAR)
# -------------------------------------------------
st.set_page_config(page_title="Hotel Room Forecast Dashboard", layout="wide")
st.title("Hotel Room Forecast Dashboard")

# ------------------------------
# SIDEBAR CONTROLS
# ------------------------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV File", type=["xlsx", "csv"])

epochs = 50
future_days = st.sidebar.slider("Forecast Days:", 7, 90, 30)

run_button = st.sidebar.button("Run Forecast for All Room Types")

# ------------------------------
# MAIN PAGE PROCESSING
# ------------------------------
if uploaded_file:
    # Read file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Convert timestamps
    df["from_date"] = pd.to_datetime(df["from_date"], unit="s", errors="coerce")
    df["to_date"] = pd.to_datetime(df["to_date"], unit="s", errors="coerce")

    # Expand room nights
    expanded_df = expand_to_room_nights(df)

    st.subheader("Expanded Room Nights Data")
    st.dataframe(expanded_df.head())

    room_types = sorted(expanded_df["room_type"].dropna().unique().tolist())

    # ------------------------------
    # RUN FORECAST BUTTON PRESSED
    # ------------------------------
    if run_button:
        all_forecasts = {}
        summary_data = []

        with st.spinner("Training and forecasting for each room type..."):
            for room in room_types:
                room_data = expanded_df[expanded_df["room_type"] == room]
                daily_data = room_data.groupby("date").agg(
                    revenue=("revenue", "sum"),
                    nights=("nights", "sum")
                ).reset_index()

                if daily_data.empty or len(daily_data) < 10:
                    continue

                max_nights = daily_data["nights"].max() if daily_data["nights"].max() > 0 else 1
                daily_data["occupancy_rate"] = (daily_data["nights"] / max_nights) * 100

                # Train models
                model_r, sc_r, n_r = lstm_forecast(daily_data["revenue"], epochs=epochs)
                model_n, sc_n, n_n = lstm_forecast(daily_data["nights"], epochs=epochs)
                model_o, sc_o, n_o = lstm_forecast(daily_data["occupancy_rate"], epochs=epochs)

                # Forecast
                future_rev = forecast_future(model_r, sc_r, daily_data["revenue"], n_r, future_days)
                future_nights = forecast_future(model_n, sc_n, daily_data["nights"], n_n, future_days)
                future_occ = forecast_future(model_o, sc_o, daily_data["occupancy_rate"], n_o, future_days)

                future_dates = pd.date_range(daily_data["date"].max() + pd.Timedelta(days=1), periods=future_days)
                forecast_df = pd.DataFrame({
                    "date": future_dates,
                    "forecast_revenue": future_rev,
                    "forecast_nights": future_nights,
                    "forecast_occupancy(%)": future_occ
                })

                all_forecasts[room] = {"historical": daily_data, "forecast": forecast_df}

                summary_data.append({
                    "Room Type": room,
                    "Last Actual Revenue": daily_data["revenue"].iloc[-1],
                    "Forecast Revenue (Next Day)": future_rev[0],
                    "Average Forecast Revenue": np.mean(future_rev)
                })

        st.success("Forecast completed for all room types!")

        # Sidebar dropdown for room selection
        selected_room = st.sidebar.selectbox("Select Room Type to Visualize Forecast", room_types)

        if selected_room in all_forecasts:
            hist = all_forecasts[selected_room]["historical"]
            fc = all_forecasts[selected_room]["forecast"]

            combined = pd.concat([
                hist.rename(columns={
                    "revenue": "actual_revenue",
                    "nights": "actual_nights",
                    "occupancy_rate": "actual_occupancy(%)"
                }),
                fc.rename(columns={
                    "forecast_revenue": "forecast_revenue",
                    "forecast_nights": "forecast_nights",
                    "forecast_occupancy(%)": "forecast_occupancy(%)"
                })
            ], ignore_index=True)

            # --------------------------------------------------------------------
            # MAIN FORECAST GRAPH
            # --------------------------------------------------------------------
            st.subheader(f"Forecast for Room Type: {selected_room}")

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(combined["date"], combined["actual_revenue"], label="Actual Revenue", color="blue")
            ax1.plot(combined["date"], combined["forecast_revenue"], label="Forecast Revenue", linestyle="--", color="orange")
            ax1.set_ylabel("Revenue (₹)", color="blue")

            ax2 = ax1.twinx()
            ax2.plot(combined["date"], combined["actual_occupancy(%)"], label="Actual Occupancy", color="green")
            ax2.plot(combined["date"], combined["forecast_occupancy(%)"], label="Forecast Occupancy", linestyle="--", color="red")
            ax2.set_ylabel("Occupancy (%)", color="green")
            st.pyplot(fig)

            st.write("Forecast Data Preview:")
            st.dataframe(combined.tail())

            # --------------------------------------------------------------------
            #  ADDITIONAL GRAPHS
            # --------------------------------------------------------------------

            # 1. BAR CHART: Actual vs Forecast Revenue
            st.subheader("Bar Chart: Actual vs Forecast Revenue")
            bar_df = pd.concat([
                hist[["date", "revenue"]].rename(columns={"revenue": "value"}).assign(type="Actual"),
                fc[["date", "forecast_revenue"]].rename(columns={"forecast_revenue": "value"}).assign(type="Forecast")
            ])
            fig_bar = px.bar(bar_df, x="date", y="value", color="type", barmode="group")
            st.plotly_chart(fig_bar, use_container_width=True)

            # 2. AREA CHART: Occupancy Trend
            st.subheader("Area Chart: Occupancy Trend")
            area_df = pd.concat([
                hist[["date", "occupancy_rate"]].rename(columns={"occupancy_rate": "value"}).assign(type="Actual"),
                fc[["date", "forecast_occupancy(%)"]].rename(columns={"forecast_occupancy(%)": "value"}).assign(type="Forecast")
            ])
            fig_area = px.area(area_df, x="date", y="value", color="type")
            st.plotly_chart(fig_area, use_container_width=True)

            # 3. DUAL AXIS: Revenue vs Nights
            st.subheader("Dual Axis: Revenue vs Nights")
            fig_dn, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(combined["date"], combined["actual_revenue"], color="blue", label="Actual Revenue")
            ax1.plot(combined["date"], combined["forecast_revenue"], color="cyan", linestyle="--", label="Forecast Revenue")
            ax1.set_ylabel("Revenue (₹)")

            ax2 = ax1.twinx()
            ax2.plot(combined["date"], combined["actual_nights"], color="green", label="Actual Nights")
            ax2.plot(combined["date"], combined["forecast_nights"], color="lime", linestyle="--", label="Forecast Nights")
            ax2.set_ylabel("Nights")
            st.pyplot(fig_dn)

            # 4. HEATMAP: Occupancy Calendar
            st.subheader("Occupancy Heatmap")
            heat_df = combined.copy()
            heat_df["day"] = heat_df["date"].dt.day
            heat_df["month"] = heat_df["date"].dt.month

            pivot_df = heat_df.pivot_table(
                index="month", columns="day", values="forecast_occupancy(%)", aggfunc="mean"
            )
            fig_heat = px.imshow(pivot_df, color_continuous_scale="Viridis")
            st.plotly_chart(fig_heat, use_container_width=True)

            # 5. SCATTER PLOT: Revenue vs Occupancy
            st.subheader("Scatter: Revenue vs Occupancy")
            scatter_df = combined.copy()
            fig_scatter = px.scatter(
                scatter_df,
                x="actual_occupancy(%)",
                y="actual_revenue",
                trendline="ols"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Summary section
        summary_df = pd.DataFrame(summary_data)
        st.subheader("Forecast Summary for All Room Types")
        st.dataframe(summary_df)

        # Download button
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            summary_df.to_excel(writer, index=False, sheet_name="Summary")
            for room, data_dict in all_forecasts.items():
                data_dict["historical"].to_excel(writer, index=False, sheet_name=f"{room[:28]}_Historical")
                data_dict["forecast"].to_excel(writer, index=False, sheet_name=f"{room[:28]}_Forecast")

        st.download_button(
            label="⬇ Download All Room Forecasts (Excel)",
            data=output.getvalue(),
            file_name="hotel_room_forecasts.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )




# # -------------------------------------------------
# # Streamlit UI (with SIDEBAR)
# # -------------------------------------------------
# st.set_page_config(page_title="Hotel Room Forecast Dashboard", layout="wide")
# st.title("Hotel Room Forecast Dashboard")

# # ------------------------------
# # SIDEBAR CONTROLS
# # ------------------------------
# st.sidebar.header("Controls")

# uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV File", type=["xlsx", "csv"])

# # epochs = st.sidebar.slider("Training Epochs:", 20, 200, 50)
# epochs = 50
# future_days = st.sidebar.slider("Forecast Days:", 7, 90, 30)

# run_button = st.sidebar.button("Run Forecast for All Room Types")

# # ------------------------------
# # MAIN PAGE PROCESSING
# # ------------------------------
# if uploaded_file:
#     # Read file
#     if uploaded_file.name.endswith(".csv"):
#         df = pd.read_csv(uploaded_file)
#     else:
#         df = pd.read_excel(uploaded_file)

#     st.subheader("📄 Raw Data Preview")
#     st.dataframe(df.head())

#     # Convert timestamps
#     df["from_date"] = pd.to_datetime(df["from_date"], unit="s", errors="coerce")
#     df["to_date"] = pd.to_datetime(df["to_date"], unit="s", errors="coerce")

#     # Expand room nights
#     expanded_df = expand_to_room_nights(df)

#     st.subheader("🛏️ Expanded Room Nights Data")
#     st.dataframe(expanded_df.head())

#     room_types = sorted(expanded_df["room_type"].dropna().unique().tolist())

#     # ------------------------------
#     # RUN FORECAST BUTTON PRESSED
#     # ------------------------------
#     if run_button:
#         all_forecasts = {}
#         summary_data = []

#         with st.spinner("Training and forecasting for each room type..."):
#             for room in room_types:
#                 room_data = expanded_df[expanded_df["room_type"] == room]
#                 daily_data = room_data.groupby("date").agg(
#                     revenue=("revenue", "sum"),
#                     nights=("nights", "sum")
#                 ).reset_index()

#                 if daily_data.empty or len(daily_data) < 10:
#                     continue

#                 max_nights = daily_data["nights"].max() if daily_data["nights"].max() > 0 else 1
#                 daily_data["occupancy_rate"] = (daily_data["nights"] / max_nights) * 100

#                 # Train models
#                 model_r, sc_r, n_r = lstm_forecast(daily_data["revenue"], epochs=epochs)
#                 model_n, sc_n, n_n = lstm_forecast(daily_data["nights"], epochs=epochs)
#                 model_o, sc_o, n_o = lstm_forecast(daily_data["occupancy_rate"], epochs=epochs)

#                 # Forecast
#                 future_rev = forecast_future(model_r, sc_r, daily_data["revenue"], n_r, future_days)
#                 future_nights = forecast_future(model_n, sc_n, daily_data["nights"], n_n, future_days)
#                 future_occ = forecast_future(model_o, sc_o, daily_data["occupancy_rate"], n_o, future_days)

#                 future_dates = pd.date_range(daily_data["date"].max() + pd.Timedelta(days=1), periods=future_days)
#                 forecast_df = pd.DataFrame({
#                     "date": future_dates,
#                     "forecast_revenue": future_rev,
#                     "forecast_nights": future_nights,
#                     "forecast_occupancy(%)": future_occ
#                 })

#                 all_forecasts[room] = {"historical": daily_data, "forecast": forecast_df}

#                 summary_data.append({
#                     "Room Type": room,
#                     "Last Actual Revenue": daily_data["revenue"].iloc[-1],
#                     "Forecast Revenue (Next Day)": future_rev[0],
#                     "Average Forecast Revenue": np.mean(future_rev)
#                 })

#         st.success("✅ Forecast completed for all room types!")

#         # Sidebar dropdown for room selection
#         selected_room = st.sidebar.selectbox("Select Room Type to Visualize Forecast", room_types)

#         if selected_room in all_forecasts:
#             hist = all_forecasts[selected_room]["historical"]
#             fc = all_forecasts[selected_room]["forecast"]

#             combined = pd.concat([
#                 hist.rename(columns={
#                     "revenue": "actual_revenue",
#                     "nights": "actual_nights",
#                     "occupancy_rate": "actual_occupancy(%)"
#                 }),
#                 fc.rename(columns={
#                     "forecast_revenue": "forecast_revenue",
#                     "forecast_nights": "forecast_nights",
#                     "forecast_occupancy(%)": "forecast_occupancy(%)"
#                 })
#             ], ignore_index=True)

#             st.subheader(f"📈 Forecast for Room Type: {selected_room}")

#             fig, ax1 = plt.subplots(figsize=(10, 5))
#             ax1.plot(combined["date"], combined["actual_revenue"], label="Actual Revenue", color="blue")
#             ax1.plot(combined["date"], combined["forecast_revenue"], label="Forecast Revenue", linestyle="--", color="orange")
#             ax1.set_ylabel("Revenue (₹)", color="blue")

#             ax2 = ax1.twinx()
#             ax2.plot(combined["date"], combined["actual_occupancy(%)"], label="Actual Occupancy", color="green")
#             ax2.plot(combined["date"], combined["forecast_occupancy(%)"], label="Forecast Occupancy", linestyle="--", color="red")
#             ax2.set_ylabel("Occupancy (%)", color="green")

#             fig.tight_layout()
#             st.pyplot(fig)

#             st.write("🔍 Forecast Data Preview:")
#             st.dataframe(combined.tail())

#         # Summary section
#         summary_df = pd.DataFrame(summary_data)
#         st.subheader("📊 Forecast Summary for All Room Types")
#         st.dataframe(summary_df)

#         # Download button
#         output = io.BytesIO()
#         with pd.ExcelWriter(output, engine="openpyxl") as writer:
#             summary_df.to_excel(writer, index=False, sheet_name="Summary")
#             for room, data_dict in all_forecasts.items():
#                 data_dict["historical"].to_excel(writer, index=False, sheet_name=f"{room[:28]}_Historical")
#                 data_dict["forecast"].to_excel(writer, index=False, sheet_name=f"{room[:28]}_Forecast")

#         st.download_button(
#             label="⬇️ Download All Room Forecasts (Excel)",
#             data=output.getvalue(),
#             file_name="hotel_room_forecasts.xlsx",
#             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#         )


# # -------------------------------------------------
# # Streamlit UI
# # -------------------------------------------------
# st.set_page_config(page_title="Hotel Room Forecast Dashboard", layout="wide")
# st.title("Hotel Room Forecast Dashboard")


# uploaded_file = st.file_uploader("Upload Excel or CSV File", type=["xlsx", "csv"])

# if uploaded_file:
#     if uploaded_file.name.endswith(".csv"):
#         df = pd.read_csv(uploaded_file)
#     else:
#         df = pd.read_excel(uploaded_file)

#     st.subheader("Raw Data Preview")
#     st.dataframe(df.head())

#     df["from_date"] = pd.to_datetime(df["from_date"], unit="s", errors="coerce")
#     df["to_date"] = pd.to_datetime(df["to_date"], unit="s", errors="coerce")

#     expanded_df = expand_to_room_nights(df)
#     st.subheader("Expanded Room Nights Data")
#     st.dataframe(expanded_df.head())

#     room_types = sorted(expanded_df["room_type"].dropna().unique().tolist())

#     # epochs = st.slider("Training Epochs:", 20, 200, 50)
#     epochs = 50
#     future_days = st.slider("Forecast Days:", 7, 90, 30)

#     if st.button("Run Forecast for All Room Types"):
#         all_forecasts = {}
#         summary_data = []

#         with st.spinner("Training and forecasting for each room type..."):
#             for room in room_types:
#                 room_data = expanded_df[expanded_df["room_type"] == room]
#                 daily_data = room_data.groupby("date").agg(
#                     revenue=("revenue", "sum"),
#                     nights=("nights", "sum")
#                 ).reset_index()

#                 if daily_data.empty or len(daily_data) < 10:
#                     continue

#                 max_nights = daily_data["nights"].max() if daily_data["nights"].max() > 0 else 1
#                 daily_data["occupancy_rate"] = (daily_data["nights"] / max_nights) * 100

#                 # Train models
#                 model_r, sc_r, n_r = lstm_forecast(daily_data["revenue"], epochs=epochs)
#                 model_n, sc_n, n_n = lstm_forecast(daily_data["nights"], epochs=epochs)
#                 model_o, sc_o, n_o = lstm_forecast(daily_data["occupancy_rate"], epochs=epochs)

#                 # Forecast future
#                 future_rev = forecast_future(model_r, sc_r, daily_data["revenue"], n_r, future_days)
#                 future_nights = forecast_future(model_n, sc_n, daily_data["nights"], n_n, future_days)
#                 future_occ = forecast_future(model_o, sc_o, daily_data["occupancy_rate"], n_o, future_days)

#                 future_dates = pd.date_range(daily_data["date"].max() + pd.Timedelta(days=1), periods=future_days)
#                 forecast_df = pd.DataFrame({
#                     "date": future_dates,
#                     "forecast_revenue": future_rev,
#                     "forecast_nights": future_nights,
#                     "forecast_occupancy(%)": future_occ
#                 })

#                 all_forecasts[room] = {"historical": daily_data, "forecast": forecast_df}

#                 summary_data.append({
#                     "Room Type": room,
#                     "Last Actual Revenue": daily_data["revenue"].iloc[-1],
#                     "Forecast Revenue (Next Day)": future_rev[0],
#                     "Average Forecast Revenue": np.mean(future_rev)
#                 })

#         st.success("Forecast completed for all room types!")

#         #  Dropdown to view forecast per room
#         selected_room = st.selectbox("Select Room Type to Visualize Forecast", room_types)

#         if selected_room in all_forecasts:
#             hist = all_forecasts[selected_room]["historical"]
#             fc = all_forecasts[selected_room]["forecast"]

#             combined = pd.concat([
#                 hist.rename(columns={
#                     "revenue": "actual_revenue",
#                     "nights": "actual_nights",
#                     "occupancy_rate": "actual_occupancy(%)"
#                 }),
#                 fc.rename(columns={
#                     "forecast_revenue": "forecast_revenue",
#                     "forecast_nights": "forecast_nights",
#                     "forecast_occupancy(%)": "forecast_occupancy(%)"
#                 })
#             ], ignore_index=True)

#             st.subheader(f"Forecast for Room Type: {selected_room}")
#             fig, ax1 = plt.subplots(figsize=(10, 5))
#             ax1.plot(combined["date"], combined["actual_revenue"], label="Actual Revenue", color="blue")
#             ax1.plot(combined["date"], combined["forecast_revenue"], label="Forecast Revenue", linestyle="--", color="orange")
#             ax1.set_ylabel("Revenue (₹)", color="blue")

#             ax2 = ax1.twinx()
#             ax2.plot(combined["date"], combined["actual_occupancy(%)"], label="Actual Occupancy", color="green")
#             ax2.plot(combined["date"], combined["forecast_occupancy(%)"], label="Forecast Occupancy", linestyle="--", color="red")
#             ax2.set_ylabel("Occupancy (%)", color="green")
#             fig.tight_layout()
#             st.pyplot(fig)

#             st.write("Forecast Data Preview:")
#             st.dataframe(combined.tail())

#         # Summary table + download
#         summary_df = pd.DataFrame(summary_data)
#         st.subheader("Forecast Summary for All Room Types")
#         st.dataframe(summary_df)

#         output = io.BytesIO()
#         with pd.ExcelWriter(output, engine="openpyxl") as writer:
#             summary_df.to_excel(writer, index=False, sheet_name="Summary")
#             for room, data_dict in all_forecasts.items():
#                 data_dict["historical"].to_excel(writer, index=False, sheet_name=f"{room[:28]}_Historical")
#                 data_dict["forecast"].to_excel(writer, index=False, sheet_name=f"{room[:28]}_Forecast")

#         st.download_button(
#             label="Download All Room Forecasts (Excel)",
#             data=output.getvalue(),
#             file_name="hotel_room_forecasts.xlsx",
#             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#         )