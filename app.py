import plotly.io as pio

import plotly.graph_objects as go
import streamlit as st
from zipfile import ZipFile

from entry import *

from pathlib import Path
import shutil

tmp_results_dir_name = 'tmp'
zip_file_name = 'results.zip'

# remove tmp if exists
tmp_path = Path(tmp_results_dir_name)
if tmp_path.exists() and tmp_path.is_dir():
    shutil.rmtree(tmp_path)
os.mkdir(tmp_results_dir_name)

st.title('Fitting Sigmoid')

uploaded_files = st.file_uploader('Upload csv file', type='csv', accept_multiple_files=True)
file_names = [f.name.split('.')[0] for f in uploaded_files]

if len(uploaded_files) > 0:
    zip_obj = ZipFile(Path(tmp_results_dir_name) / zip_file_name, 'w')

    df_raw_lst = []
    intermediate_lst = []
    final_lst = []
    aggregated_lst = []
    final_aggregated_lst = []

    for i, uploaded_file in zip(range(len(uploaded_files)), uploaded_files):
        st.write(f'processing {i+1} file out of {len(uploaded_files)}')
        df_raw = pd.read_csv(uploaded_file, sep=';')

        intermediate = pre_process(df_raw)
        final = fit_model(intermediate)
        aggregated = mean_each_trial(intermediate)
        final_aggregated = fit_model(aggregated)

        per_substrate_joined_with_model = final.merge(final_aggregated, on='trial')
        aggregated_joined_with_model = aggregated.merge(final_aggregated, on='trial')

        df_raw_lst.append(df_raw)
        intermediate_lst.append(intermediate)
        final_lst.append(final)
        aggregated_lst.append(aggregated)
        final_aggregated_lst.append(final_aggregated)

        per_substrate_file_name = f'per_substrate_{uploaded_file.name}'
        aggregated_file_name = f'final_{uploaded_file.name}'

        # intermediate.to_csv(tmp_path / f'{uploaded_file.name}_preprocessed')
        per_substrate_joined_with_model.to_csv(tmp_path / per_substrate_file_name)
        # aggregated.to_csv(tmp_path / f'{uploaded_file.name}_aggregated_per_trial')
        # final_aggregated.to_csv(tmp_path / f'{uploaded_file.name}_model_final')
        aggregated_joined_with_model.to_csv(tmp_path / aggregated_file_name)

        zip_obj.write(tmp_path / per_substrate_file_name)
        zip_obj.write(tmp_path / aggregated_file_name)

    if st.checkbox('Show raw data'):
        st.subheader('raw data')
        for df_raw, file_name in zip(df_raw_lst, file_names):
            st.write(file_name)
            st.write(df_raw)

    if st.checkbox('Show processed data'):
        st.subheader('preprocessed data')
        for intermediate, file_name in zip(intermediate_lst, file_names):
            st.write(file_name)
            st.write(intermediate)

        # st.subheader('model fitted per substrate')
        # for final, file_name in zip(final_lst, file_names):
        #     st.write(file_name)
        #     st.write(final)

        st.subheader('mean each trial')
        for aggregated, file_name in zip(aggregated_lst, file_names):
            st.write(file_name)
            st.write(aggregated)

        st.subheader('model fitted per mean')
        for final_aggregated, file_name in zip(final_aggregated_lst, file_names):
            st.write(file_name)
            st.write(final_aggregated)

    # if st.button('Show plot for aggregated data'):
    for aggregated, file_name in zip(aggregated_lst, file_names):
        st.write(file_name)
        for i in aggregated.index:
            to_plot = aggregated[aggregated.index == i].transpose()
            to_plot.index = to_plot.index.astype(float)

            hours = to_plot.index.values
            xs = np.linspace(min(hours), max(hours))
            ys_measured = to_plot.values.reshape(len(to_plot))

            params = final_aggregated.iloc[i - 1]
            L, x0, k, b = params['L'], params['x0'], params['k'], params['b']
            ys_fitted = sigmoid(xs, L, x0, k, b)

            fig = go.Figure(layout_title_text=f'trial {i}')
            fig.add_trace(trace=go.Scatter(x=hours, y=ys_measured, mode='markers', name='measured'))
            fig.add_trace(go.Scatter(x=xs, y=ys_fitted, mode='lines', name='fitted'))

            image_path = Path(tmp_results_dir_name) / f'chart_{file_name}_{i}'
            pio.write_image(fig, image_path, format='png')
            zip_obj.write(image_path)

            st.plotly_chart(fig)

    zip_obj.close()

with open(tmp_path / zip_file_name, "rb") as fp:
    btn = st.download_button(
        label="Download results",
        data=fp,
        file_name=zip_file_name,
        mime="application/zip"
    )
