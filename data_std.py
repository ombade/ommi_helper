import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer, QuantileTransformer, OrdinalEncoder
import pandas as pd

def scale_and_visualize(data, columns_to_scale):
    """
    Visualize the original data and the data after each scaling method in a 2x2 grid.

    Parameters:
    - data: DataFrame, input data to be scaled
    - columns_to_scale: list, columns to be scaled using different scalers

    Returns:
    - None
    """
    # Create instances of each scaler class
    min_max_scaler = MinMaxScaler()
    standard_scaler = StandardScaler()
    robust_scaler = RobustScaler()
    max_abs_scaler = MaxAbsScaler()
    power_transformer = PowerTransformer()
    quantile_transformer = QuantileTransformer()
    ordinal_encoder = OrdinalEncoder()

    # Copy the input data to avoid modifying the original DataFrame
    data_copy = data.copy()

    # Initialize scalers
    scalers = {
        "Original": None,
        "MinMaxScaler": min_max_scaler,
        "StandardScaler": standard_scaler,
        "RobustScaler": robust_scaler,
        "MaxAbsScaler": max_abs_scaler,
        "PowerTransformer": power_transformer,
        "QuantileTransformer": quantile_transformer,
        "OrdinalEncoder": ordinal_encoder
    }

    # Set up subplots in a 2x2 grid
    fig, axs = plt.subplots(4, 2, figsize=(20, 15))
    fig.suptitle("Comparison of Different Scalers", y=1.02)

    # Flatten the axs array for easy iteration
    axs_flat = axs.flatten()

    for i, (scaler_name, scaler) in enumerate(scalers.items()):
        axs_flat[i].set_title(scaler_name)

        if scaler is not None:
            # Scale the specified columns
            data_copy[columns_to_scale] = scaler.fit_transform(data_copy[columns_to_scale])

        # Plot KDE for all columns
        for col in columns_to_scale:
            sns.kdeplot(data_copy[col], ax=axs_flat[i], label=col)

        axs_flat[i].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()
