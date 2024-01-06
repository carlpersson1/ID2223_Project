import gradio as gr
import hopsworks
from PIL import Image

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
dataset_api = project.get_dataset_api()

# List of features:
feat = ['temperature_2m', 'apparent_temperature', 'rain', 'snowfall', 'surface_pressure', 'cloud_cover',
        'wind_speed_10m', 'wind_direction_10m']


def refresh_images():
    """Function to download the latest images from the hopsworks database."""
    # Download all the images
    for feature in feat:
        dataset_api.download('Resources/predictions' +'/pred_' + feature + '.png', overwrite=True)
        dataset_api.download('Resources/predictions' + '/prev_' + feature + '.png', overwrite=True)

    images = []

    for feature in feat:
        images.append(Image.open('pred_' + feature + '.png'))
        images.append(Image.open('prev_' + feature + '.png'))

    return images

css = """
h1 {
    text-align: center;
    display:block;
}

p {
    text-align: center;
    display:block;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown('<h1>Weather prediction service </h1>'
                '<p>These graphs shows the latest weather predictions of our model. This includes various features '
                'such as temperature, pressure, rain and so on. These predictions are based on the latest data '
                'from our dataset, which is updated every day with two day old data. The predictions are based '
                'on the weather data from stockholm specifically and the model predictions are unfortunately already '
                'outdated by the time they are made, due to the delay in data updates of our database. </p>'
                '<p>Press the update button to get the latest prediction and the performance of our last predictions '
                'using our model. The performance degrade severely after one day and as such this interval'
                'is kept for predictions. </p>')

    # Create a refresh button to download the newest data
    with gr.Row():
        gr.Column()
        with gr.Column():
            ref_btn = gr.Button('Refresh')
        gr.Column()

    # Load the initial images
    images = refresh_images()

    # Create a tab for the newest predictions (in our case 2 days old...) and for old prediction and the performance
    with gr.Tabs():
        with gr.TabItem('Latest weather prediction'):
        # Create rows with two figures each
            with gr.Row():
                with gr.Column():
                    temp = gr.Image(value=images[0], label='Temperature (°C)', show_download_button=True,
                                    interactive=False)
                with gr.Column():
                    app_temp = gr.Image(value=images[2], label='Apparent temperature (°C)', show_download_button=True,
                                        interactive=False)
            with gr.Row():
                with gr.Column():
                    rain = gr.Image(value=images[4], label='Rain (mm)', show_download_button=True, interactive=False)
                with gr.Column():
                    snow = gr.Image(value=images[6], label='snowfall (cm)', show_download_button=True,
                                    interactive=False)
            with gr.Row():
                with gr.Column():
                    press = gr.Image(value=images[8], label='Surface pressure (hPa)', show_download_button=True,
                                     interactive=False)
                with gr.Column():
                    cloud = gr.Image(value=images[10], label='cloud_cover (%)', show_download_button=True,
                                     interactive=False)
            with gr.Row():
                with gr.Column():
                    speed = gr.Image(value=images[12], label='Wind speed (km/h)', show_download_button=True,
                                     interactive=False)
                with gr.Column():
                    direction = gr.Image(value=images[14], label='Wind direction (°)', show_download_button=True,
                                         interactive=False)

        with gr.TabItem('Previous prediction performance'):
            with gr.Row():
                with gr.Column():
                    ptemp = gr.Image(value=images[1], label=r'Temperature (°C)', show_download_button=True,
                                     interactive=False)
                with gr.Column():
                    papp_temp = gr.Image(value=images[3], label='Apparent temperature (°C)', show_download_button=True,
                                         interactive=False)
            with gr.Row():
                with gr.Column():
                    prain = gr.Image(value=images[5], label='Rain (mm)', show_download_button=True, interactive=False)
                with gr.Column():
                    psnow = gr.Image(value=images[7], label='snowfall (cm)', show_download_button=True,
                                     interactive=False)
            with gr.Row():
                with gr.Column():
                    ppress = gr.Image(value=images[9], label='Surface pressure (hPa)', show_download_button=True,
                                      interactive=False)
                with gr.Column():
                    pcloud = gr.Image(value=images[11], label='cloud_cover (%)', show_download_button=True,
                                      interactive=False)
            with gr.Row():
                with gr.Column():
                    pspeed = gr.Image(value=images[13], label='Wind speed (km/h)', show_download_button=True,
                                      interactive=False)
                with gr.Column():
                    pdirection = gr.Image(value=images[15], label='Wind direction (°)', show_download_button=True,
                                          interactive=False)

    # On button click update all images:
    ref_btn.click(refresh_images, inputs=None, outputs=[temp, ptemp, app_temp, papp_temp, rain, prain, snow, psnow,
                                                        press, ppress, cloud, pcloud, speed, pspeed, direction, pdirection])

demo.launch()
