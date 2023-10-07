import plotly.graph_objects as go
import streamlit as st

from entry import *

st.title('Fitting Sigmoid')

uploaded_file = st.file_uploader('Upload csv file', type='csv')

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file, sep=';')

    intermediate = pre_process(df_raw)
    final = fit_model(intermediate)
    aggregated = mean_each_trial(intermediate)
    final_aggregated = fit_model(aggregated)

    aggregated_joined_with_model = aggregated.merge(final_aggregated, on='trial')

    if st.checkbox('Show raw data'):
        st.subheader('raw data')
        st.write(df_raw)

    if st.checkbox('Show processed data'):
        st.subheader('preprocessed data')
        st.write(intermediate)

        st.subheader('model fitted per substrate')
        st.write(final)

        st.subheader('mean each trial')
        st.write(aggregated)

        st.subheader('model fitted per mean')
        st.write(final_aggregated)

    if st.button('Show plot for aggregated data'):
        for i in aggregated.index:
            to_plot = aggregated[aggregated.index == i].transpose()
            to_plot.index = to_plot.index.astype(float)

            hours = to_plot.index.values
            xs = np.linspace(min(hours), max(hours))
            ys_measured = to_plot.values.reshape(10)

            params = final_aggregated.iloc[i-1]
            L, x0, k, b = params['L'], params['x0'], params['k'], params['b']
            ys_fitted = sigmoid(xs, L, x0, k, b)

            fig = go.Figure()
            fig.add_trace(trace=go.Scatter(x=hours, y=ys_measured, mode='markers', name='measured'))
            fig.add_trace(go.Scatter(x=xs, y=ys_fitted, mode='lines', name='fitted'))

            st.plotly_chart(fig)


    @st.cache_data
    def convert_df(df):
        return df.to_csv().encode('utf-8')


    csv = convert_df(final_aggregated)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=uploaded_file.name,
        mime='text/csv',
    )
