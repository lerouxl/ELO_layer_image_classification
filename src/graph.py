import pandas as pd
import PIL
import plotly.express as px
from src import image as im
import numpy as np

def create_pie(df: pd.DataFrame):
    """
    Create a pie chart of the class repartition
    :param df:
    :return:
    """
    pie_data = df["classification_name"].value_counts().sort_index()
    labels = pie_data.index.tolist()
    value = list(pie_data)
    pie_chart = px.pie(names=labels,
                       color=labels,
                       values=value,
                       title="Class repartition",
                       opacity=0.5,
                       color_discrete_map={
                           "powder": "black",
                           "corner": "grey",
                           "good": "green",
                           "porous": "red",
                           "bulging": "blue"
                       })
    pie_chart.update_layout(title_x=0.5)

    return pie_chart

def create_image_classification(df: pd.DataFrame, img_path: str, score: np.array, width, height, CellSize):
    """
    Create the main report
    :param score: score matrix from the model prediction
    :param img_path: Path to the layer image, will be used as background
    :param df: All the resutls
    :param width: Physical width of the image
    :param height: Physical height of the image
    :param CellSize: Size of tiles to be classified
    :return:
    """
    # Add market size parameters
    df["square size"] = 0.1
    df_copy = df
    sizes = list(range(2, 30, 1)) + list(range(30, 200, 5))
    for i in sizes:
        df_copy["square size"] = i / 10
        df = pd.concat([df, df_copy])
    # Create the raw figure with fix dimension
    fig = px.scatter(df, x="x", y="y", color="classification_name",
                     custom_data=["classification_name", "powder", "corner", "good", "porous", "bulging"],
                     animation_frame="square size",
                     size="square size",
                     size_max=100,
                     symbol_sequence=["square"],
                     opacity=0.3,
                     hover_data=["classification_name", "powder", "corner", "good", "porous", "bulging"],
                     width=900,
                     height=900,
                     color_discrete_map={
                         "powder": "black",
                         "corner": "grey",
                         "good": "green",
                         "porous": "red",
                         "bulging": "blue"},
                     )
    fig.update_layout(title_text="ELO areas classification", title_x=0.5)   # Set title
    fig.update_layout(legend_title_text='Classification:')                  # Set legend title
    # Set the legend position
    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.01,
        xanchor="center",
        x=0.5,
        orientation="h"
    ))
    image = im.get_image(img_path, width, height, CellSize)
    image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    fig.add_layout_image(
        dict(
            source=image,
            xref="x",
            yref="y",
            x=0,
            y=score.shape[2],
            sizex=score.shape[1],
            sizey=score.shape[2],
            sizing="stretch",
            opacity=1,
            layer="below"
        )
    )
    # Set the array ticks
    fig.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=0,
            dtick=1
        )
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    # Update the hover
    CUSTOM_HOVERTEMPLATE = "<br>".join([
        "Classification: <b>%{customdata[0]}</b>",
        "Powder score: %{customdata[1]}",
        "Corner score: %{customdata[2]}",
        "Good score: %{customdata[3]}",
        "Porous score: %{customdata[4]}",
        "Bulging score: %{customdata[5]}"
    ])
    fig.update_traces(
        hovertemplate=CUSTOM_HOVERTEMPLATE
    )
    for frame in fig.frames:
        for data in frame.data:
            data.hovertemplate = CUSTOM_HOVERTEMPLATE
    
    fig.update_xaxes(visible=False, showgrid=True, showticklabels=False) # Hide x axis
    fig.update_yaxes(visible=False, showgrid=True, showticklabels=False) # Hide y axis
    fig.update_layout(template="plotly_white")     # Remove layout

    return fig
