import plotly.io as pio

import plotly.graph_objects as go
import streamlit as st
from zipfile import ZipFile

from entry import *

from pathlib import Path
import shutil

st.set_page_config(layout="wide")

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

calculate_model_per_substrate = st.checkbox('Calculate model per substrate')

method = st.radio(label='Optimization algorithm', options=['lm', 'dogbox', 'trf'])

if st.button('Run'):
    zip_obj = ZipFile(Path(tmp_results_dir_name) / zip_file_name, 'w')

    df_raw_lst = []
    intermediate_lst = []
    final_lst = []
    aggregated_lst = []
    final_aggregated_lst = []

    for i, uploaded_file in zip(range(len(uploaded_files)), uploaded_files):
        st.write(f'processing {i + 1} file out of {len(uploaded_files)} ({uploaded_file.name})')

        df_raw = pd.read_csv(uploaded_file, sep=';', decimal='.')
        df_raw_lst.append(df_raw)

        intermediate = pre_process(df_raw)
        intermediate_lst.append(intermediate)

        aggregated = mean_each_trial(intermediate)
        aggregated_lst.append(aggregated)

        final_aggregated = fit_model(aggregated, method)
        final_aggregated_lst.append(final_aggregated)

        aggregated_joined_with_model = aggregated.merge(final_aggregated, on='trial')

        aggregated_file_name = f'final_{uploaded_file.name}'
        aggregated_joined_with_model.to_csv(tmp_path / aggregated_file_name)
        zip_obj.write(tmp_path / aggregated_file_name)

        if calculate_model_per_substrate:
            final = fit_model(intermediate, method)
            final_lst.append(final)

            per_substrate_joined_with_model = intermediate.merge(final, on='trial')

            per_substrate_file_name = f'per_substrate_{uploaded_file.name}'
            per_substrate_joined_with_model.to_csv(tmp_path / per_substrate_file_name)
            zip_obj.write(tmp_path / per_substrate_file_name)

    for aggregated, file_name in zip(aggregated_lst, file_names):
        st.write(file_name)
        columns = st.columns(len(aggregated.index))
        for i in aggregated.index:
            with columns[i - 1]:
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

                image_path = Path(tmp_results_dir_name) / f'chart_{file_name}_{i}.png'
                pio.write_image(fig, image_path, format='png')
                zip_obj.write(image_path)

                st.plotly_chart(fig, use_container_width=True, key=f'chart_{file_name}_{i}')

    zip_obj.close()

    with open(tmp_path / zip_file_name, "rb") as fp:
        btn = st.download_button(
            label="Download results",
            data=fp,
            file_name=zip_file_name,
            mime="application/zip"
        )
