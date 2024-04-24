import re

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import ticker

import numpy as np
import pandas as pd
import seaborn as sns
import shap
import torch
import sklearn
from scipy.stats import gaussian_kde
from sklearn.compose import ColumnTransformer


class KernelSHAP_DRN:
    """
    This class produces the Kernel SHAP values regarding the distributional property of interest.
    It produces the raw Kernel SHAP values.
    It also generates SHAP dependence plot for any pair of features, considering categorical features.
    Beeswarm plot can be generated for any features.
    """

    def __init__(
        self,
        explaining_data,
        nsamples_background_fraction,
        background_data_df_before_one_hot: pd.DataFrame,
        one_hot_encoder,
        value_function,
        glm_value_function,
        other_shap_values=None,
    ):
        """
        Args:
        See the DRNExplainer class for explanations regarding
        {explaining_data, nsamples_background_fraction, background_data_df_before_one_hot, one_hot_encoder}
        value_function: v_{M}(S, x), given any instance x and indices S \\subseteq \\{1, ..., p\\}
        """
        super(KernelSHAP_DRN, self).__init__()
        self.background_data_df_before_one_hot = background_data_df_before_one_hot
        self.value_function = value_function
        self.explaining_data = explaining_data
        self.other_shap_values = other_shap_values
        self.feature_names = self.background_data_df_before_one_hot.columns
        self.one_hot_encoder = one_hot_encoder
        sample_size = int(
            np.round(
                self.background_data_df_before_one_hot.shape[0]
                * nsamples_background_fraction
            )
        )

        if self.other_shap_values is None:
            # Compute SHAP values for the DRN network
            kernel_shap_explainer = shap.KernelExplainer(
                self.value_function,
                shap.sample(
                    self.background_data_df_before_one_hot, nsamples=sample_size
                ),
            )
            self.shap_values_kernel = kernel_shap_explainer(
                self.explaining_data, l1_reg=0.0
            )
            self.shap_base_values = self.shap_values_kernel.base_values
            self.shap_values = self.shap_values_kernel.values

            # Compute SHAP values for the GLM if required
            self.glm_value_function = glm_value_function
            if self.glm_value_function is not None:
                kernel_shap_explainer_glm = shap.KernelExplainer(
                    self.glm_value_function,
                    shap.sample(
                        self.background_data_df_before_one_hot, nsamples=sample_size
                    ),
                )
                self.shap_values_kernel_glm = kernel_shap_explainer_glm(
                    self.explaining_data, l1_reg=0.0
                )

    def forward(self):
        """
        The raw Kernel SHAP (either adjusted or DRN) output.
        """
        return self.shap_values_kernel

    def shap_glm_values(self):
        """
        The raw Kernel SHAP (GLM) output.
        """
        return self.shap_values_kernel_glm

    def shap_values_mean_adjustments(self):
        """
        The SHAP values and feature names
        """
        return (self.shap_base_values, self.shap_values, self.feature_names)

    def shap_dependence_plot(self, features_tuple, output="value"):
        """
        Create the SHAP dependence plots
        features_tuple: the pair of features required for plotting
        other_shap_values: allows for externally calculated SHAP values, i.e., FastSHAP...
        """
        tuple_indexes = [
            np.where(self.feature_names == feature)[0][0]
            for feature in features_tuple
            if feature in self.feature_names
        ]
        instances = self.explaining_data.copy()

        # Convert categorical features to numeric
        #  Encoding details
        encoders = {}
        for feature in features_tuple:
            if (
                instances[feature].dtype == object
                or instances[feature].dtype == "category"
            ):
                # not isinstance(instances[feature].values[0], float):
                encoder = sklearn.preprocessing.LabelEncoder()
                instances[feature] = encoder.fit_transform(instances[feature])
                encoders[feature] = encoder
                for class_label, encoding in zip(
                    encoder.classes_, encoder.transform(encoder.classes_)
                ):
                    print(f"{feature}: {class_label} -> {encoding}")

        if self.other_shap_values is not None:
            shap_for_tuple = self.other_shap_values
        else:
            shap_for_tuple = self.shap_value_selection(tuple_indexes, output)

        instances_for_tuple = instances.values[:, tuple_indexes]
        feature_name_tuple = self.feature_names[tuple_indexes]

        batch_size = shap_for_tuple.shape[0]

        fig, axes = plt.subplots(1, 2, figsize=(25, 10))  # 1 row, 2 columns

        for feature_idx in range(2):
            shap_values = shap_for_tuple[:, feature_idx : (feature_idx + 1)].reshape(
                batch_size, 1
            )
            feature_values = instances_for_tuple[:, feature_idx : (feature_idx + 1)]

            # Determine the color based on the other feature's value
            color_feature_index = 1 - feature_idx
            colors = instances_for_tuple[
                :, color_feature_index : (color_feature_index + 1)
            ]

            ax = axes[feature_idx]
            scatter = ax.scatter(
                feature_values, shap_values, alpha=0.85, c=colors, cmap="viridis", s=22
            )

            # Handle categorical feature for x-axis
            if feature_name_tuple[feature_idx] in encoders:
                encoder = encoders[feature_name_tuple[feature_idx]]
                ticks_and_labels = list(enumerate(encoder.classes_))
                ticks, labels = zip(*ticks_and_labels)
                ax.set_xticks(ticks)
                ax.set_xticklabels(labels)

            # Improved Handling of Categorical Feature for Color Bar
            if feature_name_tuple[color_feature_index] in encoders:
                encoder = encoders[feature_name_tuple[color_feature_index]]
                # Create a color map with a color for each unique value in the feature
                unique_vals = np.unique(instances_for_tuple[:, color_feature_index])
                cmap = mcolors.ListedColormap(
                    plt.cm.viridis(np.linspace(0, 1, len(unique_vals)))
                )
                norm = mcolors.BoundaryNorm(
                    np.arange(-0.5, len(unique_vals) + 0.5, 1), cmap.N
                )

                scatter.set_cmap(cmap)
                scatter.set_norm(norm)

                # Create the colorbar using the specified cmap and norm
                cbar = plt.colorbar(scatter, ax=ax, ticks=np.arange(len(unique_vals)))
                cbar.set_ticklabels(encoder.classes_)
                cbar.set_label(
                    f"{feature_name_tuple[color_feature_index]}", fontsize=30
                )

                # Ensuring dot-like representation for categorical color bar
                cbar.ax.minorticks_off()  # This removes any minor ticks
                cbar.ax.get_yaxis().set_major_locator(ticker.MaxNLocator(integer=True))
            else:
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label(
                    f"{feature_name_tuple[color_feature_index]}", fontsize=30
                )  # Set the font size as needed

            ax.set_xlabel(f"{feature_name_tuple[feature_idx]}", fontsize=30)
            ax.set_ylabel(f"$\\phi_{{{tuple_indexes[feature_idx]+1}}}$", fontsize=30)
            ax.set_title(
                f"SHAP Values for {feature_name_tuple[feature_idx]}", fontsize=30
            )

        # plt.tick_params(axis='x', labelsize=14)
        # plt.tick_params(axis='y', labelsize=14)
        plt.tight_layout()

    def beeswarm_plot(self, features=None, output="value"):
        """
        Create the beeswarm summary plots
        features: a list of feature names required for plotting
        adjusting: False --> explaining the drn model; True --> explaining how the drn adjusts the glm
        """
        features = (
            self.background_data_df_before_one_hot.columns
            if features is None
            else features
        )
        features_idx = [
            np.where(self.feature_names == feature)[0][0]
            for feature in features
            if feature in self.feature_names
        ]

        if self.other_shap_values is not None:
            shap_values = self.other_shap_values
        else:
            shap_values = self.shap_value_selection(features_idx, output)

        # Beeswarm summary plot
        shap.summary_plot(
            shap_values, self.explaining_data.iloc[:, features_idx], plot_size=(8, 6)
        )

    def global_importance_plot(self, features=None, output="value"):
        """
        Creates a global importance plot based on the absolute SHAP values.
        """
        features = (
            self.background_data_df_before_one_hot.columns
            if features is None
            else features
        )
        features_idx = [
            np.where(self.feature_names == feature)[0][0]
            for feature in features
            if feature in self.feature_names
        ]

        if self.other_shap_values is not None:
            shap_values = self.other_shap_values
        else:
            shap_values = self.shap_value_selection(features_idx, output)

        feature_names = self.feature_names

        # Sum the absolute SHAP values for each feature across all samples
        shap_sum = np.abs(shap_values).mean(axis=0)

        # Sort the features by their importance
        feature_importance = sorted(
            zip(feature_names, shap_sum), key=lambda x: x[1], reverse=True
        )
        sorted_features, sorted_importances = zip(*feature_importance)

        # Create the plot
        plt.figure(figsize=(15, 15))
        plt.barh(
            range(len(sorted_importances)),
            sorted_importances,
            tick_label=sorted_features,
        )
        plt.xlabel("Mean Absolute SHAP Value (Global Importance)", fontsize=35)
        plt.ylabel("Features", fontsize=35)
        plt.title("Global Feature Importance Based on SHAP Values", fontsize=35)
        plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top

    def shap_value_selection(self, indexes, output="value"):
        if output == "value":
            shap_values = self.shap_values[:, indexes]
        elif output == "glm":
            if self.glm_value_function is None:
                raise ValueError(f"Set glm_output = True while initilising the class!")
            else:
                shap_values = self.shap_values_kernel_glm.values[:, indexes]
        elif output == "drn":
            if self.glm_value_function is None:
                raise ValueError(f"Set glm_output = True while initilising the class!")
            else:
                shap_values = (
                    self.shap_values[:, indexes]
                    + self.shap_values_kernel_glm.values[:, indexes]
                )
        return shap_values


class DRNExplainer:
    def __init__(
        self,
        drn: torch.nn.Module,
        glm: torch.nn.Module,
        default_cutpoints: list,
        background_data_df_before_one_hot: pd.DataFrame,
        cat_features: list,
        all_categories: list = None,
        column_transformer: ColumnTransformer = None,
    ):
        """
        Initialise the DRNExplainer with the given parameters.

        Args:
            drn (torch.nn.Module): The DRN neural network model.
            glm (torch.nn.Module): The baseline Generalised Linear Model (GLM).
            default_cutpoints (list): Cutpoints used for training the DRN.
            background_data_df_before_one_hot (pd.DataFrame): Background data prior to one-hot encoding.
            cat_features (list): List of categorical features.
            all_categories (list, optional): List of all categories of the categorical features, if available.
            column_transformer (ColumnTransformer, optional): Preprocessor for numerical features.
        """

        super().__init__()  # Initialize the superclass. Needed if the superclass has an __init__ to run.

        # Assign the provided arguments to instance variables.
        self.drn = drn  # The DRN model.
        self.glm = glm  # The baseline GLM model.
        self.default_cutpoints = (
            default_cutpoints  # Cutpoints for feature partitioning.
        )
        self.column_transformer = (
            column_transformer  # Transformer for scaling numerical features.
        )

        # Store the raw data and features information.
        self.background_data_df_before_one_hot = (
            background_data_df_before_one_hot  # Data before one-hot encoding.
        )
        self.cat_features = cat_features  # Categorical feature names.
        self.all_categories = (
            all_categories  # Categories for all features if available.
        )

        # One-hot encode background data if categories are provided, else store it as is.
        self.background_data_df = (
            self.one_hot_encoder(self.background_data_df_before_one_hot)
            if self.all_categories is not None
            else self.background_data_df_before_one_hot
        )

        self.background_data_tensor = self._to_tensor(
            self.background_data_df
        )  # Convert the data to a PyTorch tensor.

    def plot_dp_adjustment_shap(
        self,
        instance_raw: pd.DataFrame,
        dist_property: str = "Mean",
        method: str = "Kernel",
        nsamples_background_fraction: float = 1.0,
        top_K_features: int = 3,
        adjustment: bool = True,
        other_df_models: list = None,
        model_names: list = None,
        cutpoints: list = None,
        num_interpolations: int = None,
        labelling_gap: float = 0.05,
        synthetic_data=None,
        synthetic_data_samples: int = int(1e6),
        observation=None,
        plot_baseline: bool = True,
        # Plotting parameters below:
        x_range: tuple = None,
        y_range: tuple = None,
        plot_y_label: str = None,
        plot_title: str = None,
        density_transparency: float = 1.0,
        shap_fontsize: int = 25,
        figsize: tuple = (20, 15),
        label_adjustment_factor: float = 1.0,
        legend_loc: str = "upper left",
    ):
        """
        Plot SHAP value-based adjustments with an option to include density functions.

        Args:
            instance_raw: Raw data before one-hot encoding, used for instance-specific analysis.
            dist_property: Distributional property to adjust ('Mean', 'Variance', 'Quantile').
            method: Technique for SHAP value computation ('Kernel', 'Tree', 'FastSHAP', etc.).
            nsamples_background_fraction: Fraction of background data for SHAP calculation.
            top_K_features: Number of top features based on SHAP values.
            adjustment: Whether to plot the SHAP values of the adjusted or unadjusted distributional property
            other_df_models: Other distributional forecasting models for comparison.
            model_names: Names of the other distributional forecasting models.
            cutpoints: Cutpoints for partitioning feature space, defaults to `self.default_cutpoints`.
            num_interpolations: Number of points for density interpolation, defaults to 2000.
            labelling_gap: Gap between labels in the plot for readability.
            synthetic_data, synthetic_data_samples: Synthetic data function for true density comparison and number of samples generated.
            observation: Specific observation value for vertical line plotting.
            plot_baseline: Flag to include baseline model's density plot.
            x_range, y_range: Axis ranges for the plot.
            plot_y_label, plot_title: Custom labels for the plot's axes and title.
            density_transparency: Alpha value for density plot transparency.
            shap_fontsize, figsize, label_adjustment_factor: Plot styling parameters.
            legend_loc: Location of the legend in the plot.
        """
        # Prepare data for plotting: One-hot encode the input and convert to a tensor.
        print("f", instance_raw)
        instance = self._to_tensor(self.one_hot_encoder(instance_raw))

        # Use default cutpoints unless explicitly provided.
        cutpoints = self.default_cutpoints if cutpoints is None else cutpoints

        # Set number of interpolation points for density estimation; default is 2000.
        num_interpolations = 2000 if num_interpolations is None else num_interpolations

        # Determine the range for density interpolation based on either specified x_range or the given cutpoints.
        lower_cutpoint = x_range[0] if x_range is not None else cutpoints[0]
        upper_cutpoint = x_range[1] if x_range is not None else cutpoints[-1]
        # Create a tensor of linearly spaced values between the lower and upper cutpoints.
        lst_tensor = torch.linspace(
            lower_cutpoint, upper_cutpoint, num_interpolations
        ).unsqueeze(-1)

        # Compute probability density functions (PDF) for GLM and DRN models.
        # GLM PDF is calculated from the log probabilities of the distribution.
        glm_pdf = np.exp(
            (self.glm.distributions(instance).log_prob(lst_tensor)).detach().numpy()
        )

        # DRN PDF is calculated from the probabilities of the distribution.
        drn_pdf = self.drn.distributions(instance).prob(lst_tensor).detach().numpy()

        # Setup the plotting environment with specified figure size.
        figure, axes = plt.subplots(1, 1, figsize=figsize)

        # Calculate the maximum Y-axis value for the plot
        y_max = 1.25 * max(drn_pdf.max(), glm_pdf[1:].max())
        if y_range is not None:
            y_min, y_max = y_range

        # Define the X-axis range
        x_min, x_max = 0, cutpoints[-1]

        # Find indices of cutpoints closest to the default cutpoints
        index_of_closest_last = torch.argmin(
            torch.abs(lst_tensor - upper_cutpoint)
        ).item()
        index_of_closest_first = torch.argmin(
            torch.abs(lst_tensor - lower_cutpoint)
        ).item()

        # Baseline Density Plotting
        if plot_baseline:
            plt.plot(
                lst_tensor[index_of_closest_first:index_of_closest_last],
                glm_pdf[index_of_closest_first:index_of_closest_last],
                color="black",
                linewidth=3,
                label="Baseline Density",
                alpha=density_transparency,
            )

        drn_pdf_value = 0
        # DRN Density Plotting
        for i in range(index_of_closest_first, index_of_closest_last - 1):
            start, end = lst_tensor[i], lst_tensor[i + 1]
            grid_new = np.linspace(start, end, 3).flatten()
            drn_pdf_value = drn_pdf[i]
            # Plot the horizontal line with increased thickness
            plt.plot(
                grid_new,
                np.full_like(grid_new, drn_pdf_value),
                color="blue",
                linewidth=3,
                alpha=density_transparency,
            )
            # Draw vertical dashed lines between the jumps
            if (
                i < index_of_closest_last - 2
            ):  # Check to avoid adding a vertical line at the end
                plt.vlines(
                    end,
                    drn_pdf[i],
                    drn_pdf[i + 1],
                    color="blue",
                    linewidth=2,
                    linestyles="dashed",
                    alpha=density_transparency,
                )

        # Add a label and plot DRN density if drn_pdf_value is greater than zero
        if drn_pdf_value > 0:
            plt.plot(
                grid_new,
                np.full_like(grid_new, drn_pdf_value),
                color="blue",
                linewidth=3,
                label="DRN Density",
                alpha=density_transparency,
            )

        # Integrate SHAP explanations into the plot
        self.kernel_shap_plot(
            instance_raw=instance_raw,
            instance=instance,
            dist_property=dist_property,
            method=method,
            nsamples_background_fraction=nsamples_background_fraction,
            adjustment=adjustment,
            axes=axes,
            top_K_features=top_K_features,
            y_max=y_max,
            y_min=y_min,
            labelling_gap=labelling_gap,
            fontsize=shap_fontsize,
        )

        # Plot synthetic data density or vertical lines to denote true values if synthetic data is provided
        if synthetic_data is not None:
            instance_np = instance.detach().numpy()[0]
            y = np.array(
                synthetic_data(
                    synthetic_data_samples, 0, specific_instance=instance_np
                )[1].values
            )
            print(instance, instance_np)
            if observation:
                # Plot vertical line for the observed property mean or quantile
                property_value = (
                    np.mean(y) if dist_property == "Mean" else np.percentile(y, 90)
                )
                axes.axvline(
                    property_value,
                    ls="dashed",
                    alpha=0.75,
                    linewidth=4,
                    label=f"True {dist_property}: {round(float(property_value), 3)}",
                    color="orange",
                )
            else:
                # Plot KDE for the synthetic data
                sns.kdeplot(
                    y,
                    color="orange",
                    label="True Density",
                    gridsize=3000,
                    linewidth=3,
                    alpha=density_transparency,
                )

        # Plot observed value if synthetic data is not available
        if observation is not None and synthetic_data is None:
            axes.axvline(
                observation,
                ls="dashed",
                alpha=0.75,
                linewidth=4,
                label=f"Observation:\n{round(float(observation), 3)}",
                color="red",
            )

        # Plot density functions from other distribution forecasting models
        if other_df_models is not None:
            for idx, DF_model in enumerate(other_df_models):
                current_pdf = np.exp(
                    DF_model.distributions(instance)
                    .log_prob(lst_tensor)
                    .detach()
                    .numpy()
                )
                plt.plot(
                    lst_tensor,
                    current_pdf,
                    label=f"{model_names[idx]} Density",
                    color=str(0.5 + 0.5 * idx / len(other_df_models)),
                    alpha=density_transparency,
                )

        # Set axis labels, title, and legend
        plt.xlabel("Y", fontsize=42)
        plt.gca().set_ylabel("")
        if plot_y_label is not None:
            plt.ylabel(plot_y_label, fontsize=42 * label_adjustment_factor)
        if plot_title is not None:
            plt.title(plot_title, fontsize=48 * label_adjustment_factor)
        else:
            plt.title(
                f'{dist_property} {"Adjustment" if adjustment else "Explanation"}',
                fontsize=48 * label_adjustment_factor,
            )

        plt.legend(prop={"size": 32 * (label_adjustment_factor - 0.2)}, loc=legend_loc)

        # Set plot ranges
        if x_range is not None:
            x_min, x_max = x_range
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(0, y_max)

    def plot_adjustment_factors(
        self,
        instance,
        cutpoints=None,
        num_interpolations=None,
        other_df_models=None,
        model_names=None,
        percentiles=None,
        cutpoints_label_bool=False,
        synthetic_data=None,
        plot_adjustments_labels=True,
        axes: plt.Axes = None,
        x_range=None,
        y_range=None,
        plot_title=None,
        plot_mean_adjustment=False,
        plot_y_label=None,
        density_transparency=1.0,
        figsize=None,
    ):
        """
        Plot the adjustment factors for each of the partitioned interval.
        expand: interpolation of cutpoints for density evaluations.
        """

        # Default cutpoints and num_interpolations setup
        cutpoints = self.default_cutpoints if cutpoints is None else cutpoints
        c_0 = self.default_cutpoints[0]
        c_K = self.default_cutpoints[-1]
        instance = (
            self._to_tensor(self.one_hot_encoder(instance))
            if self.all_categories is not None
            else self._to_tensor(instance)
        )

        # If we are plotting concerning quantiles
        if percentiles is not None:
            lst_tensor = torch.linspace(
                cutpoints[0], cutpoints[-1], num_interpolations
            ).unsqueeze(-1)
            print(lst_tensor[:5])
            cutpoints_label_bool = True
            print(self.glm.quantiles(instance, percentiles).detach().numpy())
            cutpoints = (
                self.glm.quantiles(instance, percentiles).detach().numpy().squeeze(1)
            )

            cutpoints[0] = cutpoints[0] + 1e-3
            print(cutpoints)

        # Interpolation
        lower_cutpoint = x_range[0] if x_range is not None else cutpoints[0]
        upper_cutpoint = x_range[1] if x_range is not None else cutpoints[-1]
        lst_tensor = torch.linspace(
            lower_cutpoint, upper_cutpoint, num_interpolations
        ).unsqueeze(-1)

        # Calculate GLM and DRN PDFs
        glm_pdf = np.exp(
            self.glm.distributions(instance).log_prob(lst_tensor).detach().numpy()
        )
        drn_pdf = self.drn.distributions(instance).prob(lst_tensor).detach().numpy()

        # Set up the plot
        if axes is not None and figsize is not None:
            axes.figure.set_size_inches(figsize)
        if axes is None:
            _, axes = plt.subplots(1, 1, figsize=figsize)

        plt.sca(axes)

        y_max = 1.5 * max(drn_pdf.max(), glm_pdf[1:].max())
        x_min, x_max = cutpoints[0], cutpoints[-1]

        # Find indices of cutpoints closest to the default cutpoints
        index_of_closest_last = torch.argmin(
            torch.abs(lst_tensor - upper_cutpoint)
        ).item()
        index_of_closest_first = torch.argmin(
            torch.abs(lst_tensor - lower_cutpoint)
        ).item()

        # Baseline Density Plotting
        plt.plot(lst_tensor, glm_pdf, color="blue", alpha=density_transparency)
        plt.plot(
            lst_tensor[index_of_closest_first:index_of_closest_last],
            glm_pdf[index_of_closest_first:index_of_closest_last],
            color="black",
            linewidth=2,
            label="GLM",
            alpha=density_transparency,
        )

        drn_pdf_value = 0
        # DRN Density Plotting
        for i in range(index_of_closest_first, index_of_closest_last - 1):
            start, end = lst_tensor[i], lst_tensor[i + 1]
            grid_new = np.linspace(start, end, 3).flatten()
            drn_pdf_value = drn_pdf[i]
            # Plot the horizontal line with increased thickness
            plt.plot(
                grid_new,
                np.full_like(grid_new, drn_pdf_value),
                color="blue",
                linewidth=2,
                alpha=density_transparency,
            )
            # Draw vertical dashed lines between the jumps
            if (
                i < index_of_closest_last - 2
            ):  # Check to avoid adding a vertical line at the end
                plt.vlines(
                    end,
                    drn_pdf[i],
                    drn_pdf[i + 1],
                    color="blue",
                    linewidth=1,
                    linestyles="dotted",
                    alpha=density_transparency,
                )

        # Add the label
        if drn_pdf_value > 0:
            plt.plot(
                grid_new,
                np.full_like(grid_new, drn_pdf_value),
                color="blue",
                linewidth=2,
                label="DRN",
                alpha=density_transparency,
            )

        # Other DF Models Plotting
        if other_df_models is not None:
            for idx, DF_model in enumerate(other_df_models):
                current_pdf = np.exp(
                    DF_model.distributions(instance)
                    .log_prob(lst_tensor)
                    .detach()
                    .numpy()
                )
                plt.plot(
                    lst_tensor,
                    current_pdf,
                    label=f"{model_names[idx]} Density",
                    color=str(0.5 + 0.5 * idx / len(other_df_models)),
                    alpha=density_transparency,
                )

        if plot_adjustments_labels:
            if cutpoints_label_bool:
                for i in range(len(cutpoints)):
                    if cutpoints[i] <= c_0 and cutpoints[i + 1] > c_0:
                        plt.text(
                            c_0,
                            y_max / 35,
                            "Adjusted $c_{0}$",
                            ha="center",
                            fontsize=14,
                            color="black",
                        )
                        plt.axvline(c_0, ls="--", color="black", alpha=0.5)
                    elif (
                        cutpoints[i] >= lst_tensor[index_of_closest_first]
                        and cutpoints[i] <= lst_tensor[index_of_closest_last]
                    ):
                        plt.text(
                            cutpoints[i],
                            y_max * 0.8,
                            f"Baseline\n {percentiles[i]}% Quantile",
                            ha="center",
                            fontsize=20,
                            color="black",
                        )
                        plt.axvline(cutpoints[i], ls="--", color="black", alpha=0.5)
                    elif cutpoints[i] >= c_K:
                        plt.text(
                            c_K,
                            y_max / 35,
                            "Adjusted $c_{K}$",
                            ha="center",
                            fontsize=14,
                            color="black",
                        )
                        plt.axvline(c_K, ls="--", color="black", alpha=0.5)
                        break

            interval_width = int(num_interpolations / len(cutpoints))
            adjustment_idx = 1
            for i in range(len(cutpoints) - 1):
                if (
                    cutpoints[-1] <= self.default_cutpoints[0]
                    or cutpoints[0] >= self.default_cutpoints[-1]
                ):
                    region_start = cutpoints[0]
                    region_end = cutpoints[-1]
                    if not cutpoints_label_bool:
                        plt.axvline(region_end, ls="--", color="black", alpha=0.5)

                    plt.text(
                        (region_start + region_end) / 2,
                        y_max / 2,
                        f"No Adjustment",
                        ha="center",
                        fontsize=16,
                        color="black",
                    )
                    break

                # cutpoint_{i} < c_0 < cutpoint_{i +1}
                elif (
                    cutpoints[i] < self.default_cutpoints[0]
                    and cutpoints[i + 1] >= self.default_cutpoints[0]
                ):
                    # we need to break it down into [cutpoint_0, c_0) and [c_0, cutpoint_{i+1})
                    # [cutpoint_0, c_0) should be labelled as "no adjustment"
                    region_start = cutpoints[0]
                    region_end = self.default_cutpoints[0]
                    if not cutpoints_label_bool:
                        plt.axvline(region_end, ls="--", color="black", alpha=0.5)
                    plt.text(
                        (region_start + region_end) / 2,
                        y_max / 2,
                        f"No Adjustment",
                        ha="center",
                        fontsize=16,
                        color="black",
                    )

                    if cutpoints[i + 1] <= self.default_cutpoints[-1]:
                        # [c_0, cutpoint_{i+1})
                        adjustment_idx = self.region_text(
                            instance,
                            interval_width,
                            drn_pdf,
                            glm_pdf,
                            y_max,
                            region_start=self.default_cutpoints[0],
                            region_end=cutpoints[i + 1],
                            cutpoint_idx=i,
                            adjustment_idx=1,
                            cutpoints_label_bool=cutpoints_label_bool,
                            percentiles=percentiles,
                        )
                    else:
                        region_start = self.default_cutpoints[0]
                        region_end = self.default_cutpoints[-1]
                        if not cutpoints_label_bool:
                            plt.axvline(region_end, ls="--", color="black", alpha=0.5)
                        plt.text(
                            (region_start + region_end) / 2,
                            y_max / 2,
                            f"No Adjustment",
                            ha="center",
                            fontsize=16,
                            color="black",
                        )

                        region_start = self.default_cutpoints[-1]
                        region_end = cutpoints[i + 1]
                        plt.text(
                            (region_start + region_end) / 2,
                            y_max / 2,
                            f"No Adjustment",
                            ha="center",
                            fontsize=16,
                            color="black",
                        )
                        break

                # cutpoint_{i} < c_K <= cutpoints_{i+1}
                elif (
                    cutpoints[i + 1] >= self.default_cutpoints[-1]
                    and cutpoints[i] >= self.default_cutpoints[0]
                ):
                    # we need to break it down into [cutpoint_{i}, c_K) and [c_K, cutpoint_{i+1})
                    adjustment_idx = self.region_text(
                        instance,
                        interval_width,
                        drn_pdf,
                        glm_pdf,
                        y_max,
                        cutpoints[i],
                        self.default_cutpoints[-1],
                        i,
                        adjustment_idx,
                        cutpoints_label_bool=cutpoints_label_bool,
                        percentiles=percentiles,
                    )

                    region_start = self.default_cutpoints[-1]
                    region_end = cutpoints[-1]
                    if region_end != region_start:
                        plt.text(
                            (region_start + region_end) / 2,
                            y_max / 2,
                            f"No Adjustment",
                            ha="center",
                            fontsize=16,
                            color="black",
                        )
                    break

                # c_0 < cutpoint_{i} < cutpoints_{i+1} < c_K
                elif (
                    cutpoints[i + 1] < self.default_cutpoints[-1]
                    and cutpoints[i] >= self.default_cutpoints[0]
                    and cutpoints[i] < self.default_cutpoints[-1]
                ):

                    adjustment_idx = self.region_text(
                        instance,
                        interval_width,
                        drn_pdf,
                        glm_pdf,
                        y_max,
                        cutpoints[i],
                        cutpoints[i + 1],
                        i,
                        adjustment_idx,
                        cutpoints_label_bool=cutpoints_label_bool,
                        percentiles=percentiles,
                    )

        if synthetic_data is not None:
            instance_np = instance.detach().numpy()[0]
            y = np.array(
                synthetic_data(500000, 0, specific_instance=instance_np)[1].values
            )
            density = gaussian_kde(y)

            # Plotting
            x = np.linspace(x_range[0], x_range[1], num_interpolations)
            plt.plot(
                x,
                density(x),
                color="red",
                linewidth=2,
                alpha=density_transparency,
                label="True (KDE)",
            )

        if plot_mean_adjustment:
            plt.ylim(bottom=0)
            DP_glm = self.glm.mean(instance).item()
            DP_drn = self.drn.distributions(instance).mean.item()

            axes.axvline(
                DP_glm,
                ls="dashdot",
                alpha=0.9,
                linewidth=4,
                label=f"Baseline Mean:\n {round(float(DP_glm), 3)}",
                color="black",
            )
            axes.axvline(
                DP_drn,
                ls="dashdot",
                alpha=0.9,
                linewidth=4,
                label=f"Adjusted Mean:\n {round(float(DP_drn), 3)}",
                color="blue",
            )

        if synthetic_data is not None and plot_mean_adjustment:
            axes.axvline(
                np.mean(y),
                ls="dashdot",
                alpha=0.9,
                linewidth=4,
                label=f"True Mean:\n {round(float(np.mean(y)))}",
                color="red",
            )

        # Adding labels and title
        plt.xlabel("$y$")

        y_label = "$f(y|X=x^*)$"
        plt.gca().set_ylabel("")
        if plot_y_label is not None:
            y_label = plot_y_label
            plt.ylabel(y_label)

        title_name = (
            "Density Plot with Adjustment Factors"
            if plot_adjustments_labels
            else "Density Plot"
        )
        if plot_title is not None:
            title_name = plot_title
        plt.title(title_name)

        plt.legend(loc="upper right")
        if x_range is not None:
            x_min, x_max = x_range
        axes.set_xlim(x_min, x_max)
        y_max = y_max if plot_adjustments_labels else None
        if y_range is not None:
            y_max = y_range[1]
        axes.set_ylim(0, y_max)

    def cdf_plot(
        self,
        instance,
        grid=None,
        cutpoints=None,
        other_df_models=None,
        model_names=None,
        synthetic_data=None,
        x_range=None,
        plot_title=None,
        plot_baseline=True,
        density_transparency=1.0,
        dist_property="Mean",
        nsamples_background_fraction=0.1,
        adjustment=True,
        method="Kernel",
        labelling_gap=0.01,
        top_K_features=3,
        y_range=None,
        shap_fontsize=25,
        figsize=(20, 15),
        label_adjustment_factor=1.0,
    ):
        """
        Plot the cumulative distribution function.
        """

        # Default cutpoints and num_interpolations setup
        cutpoints = self.default_cutpoints if cutpoints is None else cutpoints
        instance_raw = instance
        instance = (
            self._to_tensor(self.one_hot_encoder(instance))
            if self.all_categories is not None
            else self._to_tensor(instance)
        )
        # Interpolation
        lower_cutpoint = x_range[0] if x_range is not None else cutpoints[0]
        upper_cutpoint = x_range[1] if x_range is not None else cutpoints[-1]
        grid = (
            torch.linspace(lower_cutpoint, upper_cutpoint, 100).unsqueeze(-1)
            if grid is None
            else grid
        )

        # Assuming 'cdf_drn' computation is done as you've mentioned
        cdfs_glm = self.glm.distributions(instance).cdf(grid).detach().numpy()
        cdfs_drn = self.drn.distributions(instance).cdf(grid).detach().numpy()

        # Plotting
        figure, axes = plt.subplots(1, 1, figsize=figsize)

        y_min = y_range[0] if y_range is not None else 1.0
        y_max = y_range[1] if y_range is not None else 0.0
        x_min, x_max = self.kernel_shap_plot(
            instance_raw=instance_raw,
            instance=instance,
            dist_property=dist_property,
            method=method,
            nsamples_background_fraction=nsamples_background_fraction,
            adjustment=adjustment,
            axes=axes,
            top_K_features=top_K_features,
            y_max=y_max,
            y_min=y_min,
            labelling_gap=labelling_gap,
            fontsize=shap_fontsize,
        )

        # Other DF Models Plotting
        if other_df_models is not None:
            for idx, DF_model in enumerate(other_df_models):
                current_cdf = (
                    DF_model.distributions(instance).cdf(grid).detach().numpy()
                )
                plt.plot(
                    grid,
                    current_cdf,
                    label=f"{model_names[idx]} CDF",
                    color=str(0.5 + 0.5 * idx / len(other_df_models)),
                    alpha=density_transparency,
                )

        # cdfs_mdn = mdn.distributions(torch.Tensor([0.5, 0.5]).reshape(1,2)).cdf(grid).detach().numpy()
        if synthetic_data is not None:
            samples = synthetic_data(
                n=500000, seed=1, specific_instance=instance.detach().numpy()[0]
            )[1].values
            if dist_property == "Mean":
                axes.axvline(
                    np.mean(samples),
                    ls="dashed",
                    alpha=0.75,
                    linewidth=4,
                    label=f"True {dist_property}:{round(float(np.mean(samples)), 3)}",
                    color="orange",
                )
            else:
                axes.axvline(
                    np.percentile(samples, 90),
                    ls="dashed",
                    alpha=0.75,
                    linewidth=4,
                    label=f"True {dist_property}:{round(float(np.percentile(samples, 90)), 3)}",
                    color="orange",
                )

        plt.plot(grid, cdfs_drn, label="DRN CDF", color="blue", linewidth=3)
        if plot_baseline:
            plt.plot(grid, cdfs_glm, label="GLM CDF", color="black", linewidth=3)
        if synthetic_data is not None:
            cdf_empirical = [
                self.empirical_cdf(samples, x) for x in grid.detach().numpy()
            ]
            plt.plot(grid, cdf_empirical, label="True CDF", color="orange", linewidth=3)

        x_min = lower_cutpoint if x_range is not None else x_min
        x_max = upper_cutpoint if x_range is not None else x_max
        axes.set_xlim(max(0, x_min), x_max)
        if y_range is not None:
            axes.set_ylim(y_range[0], y_range[1])
        plt.xlabel("$y$", fontsize=42 * label_adjustment_factor)
        plt.ylabel("Cumulative Probability", fontsize=32 * label_adjustment_factor)

        plt.gca().set_ylabel("")
        if plot_title is not None:
            title_name = plot_title
            plt.title(title_name, fontsize=48 * label_adjustment_factor)

        # plt.title('Cumulative Distribution Functions (CDFs)', fontsize = 36)
        plt.legend(prop={"size": 32 * (label_adjustment_factor - 0.15)})

    def kernel_shap_plot(
        self,
        instance_raw: pd.DataFrame,
        instance: torch.Tensor,
        dist_property: str,
        method="Kernel",
        nsamples_background_fraction: float = 1.0,
        adjustment: bool = True,
        axes=None,
        top_K_features: int = 3,
        # Plot styling parameters grouped at the end
        y_max: float = None,
        y_min: float = None,
        labelling_gap: float = 0.05,
        fontsize: int = 25,
    ):
        """
        Visualises the impact of SHAP values for the top K features on a specified distributional property,
        including the option to display adjustment effects.

        Parameters:
        - instance_raw (pd.DataFrame): The instance data prior to any processing.
        - instance (torch.Tensor): Processed instance data ready for the model.
        - dist_property (str): Target distributional property (e.g., 'Mean', 'Variance').
        - method: Method used for SHAP value computation, defaulting to 'Kernel'.
        - nsamples_background_fraction (float): Fraction of background data utilised, defaults to 1.0.
        - adjustment (bool): Whether to include adjustment effects in the visualisation, defaults to True.
        - axes: Matplotlib axes object for plotting.
        - top_K_features (int): Number of top features to highlight based on SHAP values.
        - Plot styling parameters like `y_max`, `y_min`, `labelling_gap`, `fontsize` are for visual adjustments.
        """

        # Compute SHAP values and model predictions
        if method == "Kernel":
            shap_base_value_diff, shap_values_diff, feature_names = self.kernel_shap(
                explaining_data=instance_raw,
                distributional_property=dist_property,
                nsamples_background_fraction=nsamples_background_fraction,
                adjustment=adjustment,
            ).shap_values_mean_adjustments()

            # Calculate distributional property values
            DP_glm, DP_drn = self._compute_distributional_properties(
                instance, dist_property, adjustment
            )
        else:
            # Placeholder for extending method functionality
            raise NotImplementedError(f"Method {method} is not implemented.")

        phi_j_s = np.concatenate((shap_base_value_diff, shap_values_diff[0]))

        # Get top K indices based on SHAP values
        top_K_indices = np.argsort(-np.abs(phi_j_s))[:top_K_features]

        # Extend feature names and instance values arrays
        feature_names = np.insert(feature_names, 0, "X_0")
        # instance_values = np.insert(instance.detach().numpy()[0], 0, 1.000)
        instance_values = np.insert(np.array(instance_raw)[0], 0, 1.000)

        # Select top K SHAP values, feature names, and instance values
        phi_j_s_top = phi_j_s[top_K_indices]
        feature_names_top = feature_names[top_K_indices]
        instance_values_top = instance_values[top_K_indices]

        # Create a boolean mask for all indices
        all_indices = np.arange(len(phi_j_s))
        mask = ~np.isin(all_indices, top_K_indices)

        # Get the remaining SHAP values
        phi_j_s_remaining = phi_j_s[mask]

        if adjustment:
            axes.axvline(
                DP_glm,
                ls="dashed",
                alpha=0.75,
                linewidth=4,
                label=f"Baseline {dist_property}:\n{round(float(DP_glm), 3)}",
                color="black",
            )

        # Plot vertical lines for means
        axes.axvline(
            DP_drn,
            ls="dashed",
            alpha=0.75,
            linewidth=4,
            label=f"DRN {dist_property}:\n{round(float(DP_drn), 3)}",
            color="blue",
        )

        # Determine x-axis range
        gap = abs(DP_glm - DP_drn) if adjustment else DP_drn / 2
        x_min, x_max = min(DP_glm, DP_drn) - 2 * gap, max(DP_glm, DP_drn) + 2 * gap

        # Plot SHAP value arrows and text
        y_position = y_max / 5
        if y_min is not None:
            y_position = y_min + (y_max - y_min) / 5
            y_max = y_max - y_min

        # For top K features
        for idx, phi in enumerate(phi_j_s_top):
            color = "green" if phi > 0 else "red"
            phi_sum_old = DP_glm + np.sum(phi_j_s_top[:idx])
            phi_sum_next = DP_glm + np.sum(phi_j_s_top[: (idx + 1)])

            y_position += y_max * labelling_gap

            if idx < (len(phi_j_s_top) - 1):
                plt.plot(
                    [phi_sum_next, phi_sum_next],
                    [y_position, y_position + y_max * labelling_gap],
                    ls="--",
                    color=color,
                    alpha=0.8,
                )
            # Draw the arrows
            axes.arrow(
                phi_sum_old,
                y_position,
                phi,
                0,
                head_width=y_max / 20,
                head_length=gap / 15,
                fc=color,
                ec=color,
                length_includes_head=True,
            )

            # Text the SHAP value
            instance_value = instance_values_top[idx]

            if not isinstance(instance_value, float):
                feature_name_text = str(instance_value)
            else:
                feature_name_text = f"{round(float(instance_value), 3)} "
            plt.text(
                phi_sum_old,
                y_position + (y_max / (50)),
                f"$\\phi_{{{top_K_indices[idx]}}}={round(float(phi), 3)}$; {feature_names_top[idx]}$= {feature_name_text}$",
                ha="left",
                fontsize=fontsize,
                color=color,
            )

            # Adjust the regions
            x_max = phi_sum_next + 2 * gap if phi_sum_next > x_max else x_max
            x_min = phi_sum_next - 2 * gap if phi_sum_next < x_min else x_min

        # Remaining features
        if len(phi_j_s_remaining) > 0:
            phi_remaining_sum = np.sum(phi_j_s_remaining)

            color = "green" if phi_remaining_sum > 0 else "red"

            plt.plot(
                [phi_sum_next, phi_sum_next],
                [y_position, y_position + y_max * labelling_gap],
                ls="--",
                color=color,
                alpha=0.8,
            )

            y_position += y_max * labelling_gap

            axes.arrow(
                phi_sum_next,
                y_position,
                phi_remaining_sum,
                0,
                head_width=y_max / 20,
                head_length=gap / 15,
                fc=color,
                ec=color,
                length_includes_head=True,
            )
            plt.text(
                phi_sum_next,
                y_position + (y_max / (50)),
                f"Remaining Features Contribute {round(float(phi_remaining_sum), 3)}",
                ha="left",
                fontsize=fontsize,
                color=color,
            )

        return (x_min, x_max)

    def _compute_distributional_properties(self, instance, dist_property, adjustment):
        """
        Computes the specified distributional properties for both GLM and DRN models.
        """
        if dist_property == "Mean":
            DP_glm = self.glm.mean(instance).detach().numpy()[0] if adjustment else 0
            DP_drn = self.drn.distributions(instance).mean.detach().numpy()
        elif dist_property == "Variance":
            DP_glm = (
                self.glm.variance(instance).detach().numpy()[0] if adjustment else 0
            )
            DP_drn = self.drn.distributions(instance).variance().detach().numpy()
        else:
            percentile = (
                int(re.search(r"(\d+)% Quantile", dist_property).group(1))
                if re.search(r"(\d+)% Quantile", dist_property)
                else None
            )
            DP_glm = (
                self.glm.quantiles(instance, [percentile]).item()
                if percentile and adjustment
                else 0
            )
            DP_drn = (
                self.drn.distributions(instance).quantiles([percentile]).item()
                if percentile
                else None
            )
        return float(DP_glm), float(DP_drn)

    def empirical_cdf(self, samples, x):
        """
        Compute the empirical CDF for a given value x based on the provided samples.
        """
        return np.mean(samples <= x)

    def kernel_shap(
        self,
        explaining_data,
        distributional_property,
        adjustment=True,
        nsamples_background_fraction=1.0,
        glm_output=False,
        other_shap_values=None,
    ):
        """
        Pass on the explaining instance, background data, feature processing and value function to the KernelSHAP_DRN class
        """

        if distributional_property == "Mean":
            value_function = lambda instances: self.mean_value_function(
                instances, adjustment=adjustment
            )
            glm_value_function = self.mean_glm
        elif distributional_property == "Variance":
            value_function = lambda instances: self.variance_value_function(
                instances, adjustment=adjustment
            )
            glm_value_function = self.variance_glm
        else:
            value_function = self.set_value_function(
                distributional_property, adjustment, self.quantile_value_function
            )
            # glm_value_function = self.set_value_function(distributional_property, adjustment, self.quantile_glm)
            match = re.search(r"(\d+)% Quantile", distributional_property)
            if match:
                # Extract the percentage as an integer
                percentile = int(match.group(1))
                glm_value_function = lambda instances: self.quantile_glm(
                    instances, percentile=[percentile]
                )

            else:
                raise ValueError(
                    "Invalid distributional property! What Are We Explaining?"
                )

        return KernelSHAP_DRN(
            explaining_data,
            nsamples_background_fraction,
            self.background_data_df_before_one_hot,
            self.one_hot_encoder,
            value_function,
            glm_value_function if glm_output else None,
            other_shap_values,
        )

    def max_pdf_in_region(self, drn_pdf, glm_pdf, interval_width, cutpoint_idx):
        """
        Find the maximum pdf value within the region
        """
        drn_pdf_value = torch.max(
            torch.Tensor(
                drn_pdf[
                    cutpoint_idx * interval_width : (cutpoint_idx + 1) * interval_width
                ]
            )
        )
        glm_pdf_value = torch.max(
            torch.Tensor(
                glm_pdf[
                    cutpoint_idx * interval_width : (cutpoint_idx + 1) * interval_width
                ]
            )
        )
        max_of_two = torch.max(drn_pdf_value, glm_pdf_value)

        return max_of_two

    def region_adjustments(self, instance, region_start, region_end):
        """
        Calculate and round the adjustment factors
        """
        region_end = (
            region_end - 1e-03
            if region_end == self.default_cutpoints[-1]
            else region_end
        )
        factor = (
            self.real_adjustment_factors(instance, [region_start, region_end])
            .detach()
            .numpy()
        )
        factor_background = np.mean(
            self.real_adjustment_factors(
                self.background_data_tensor, [region_start, region_end]
            )
            .detach()
            .numpy(),
            axis=1,
        )
        rounded_factor = round(float(factor[0][0]), 2)
        rounded_factor_background = round(float(factor_background[0]), 2)

        return (rounded_factor, rounded_factor_background)

    def region_text(
        self,
        instance,
        interval_width,
        drn_pdf,
        glm_pdf,
        y_max,
        region_start,
        region_end,
        cutpoint_idx,
        adjustment_idx,
        cutpoints_label_bool=False,
        percentiles=None,
    ):
        """
        Text the density adjustment regions
        """
        max_of_two = self.max_pdf_in_region(
            drn_pdf, glm_pdf, interval_width, cutpoint_idx
        )
        rounded_factor, rounded_factor_background = self.region_adjustments(
            instance, region_start, region_end
        )

        if region_start != region_end:
            text_label = f"$\\hat{{a}}_{{{adjustment_idx}}}={rounded_factor}$"
            # text_label_background = f'Avg.$={rounded_factor_background}$'
            plt.text(
                float(region_start + region_end) / 2,
                max_of_two + y_max / 15,
                text_label,
                ha="center",
                fontsize=20,
                color="blue",
            )
            # plt.text(float(region_start + region_end) / 2,  max_of_two-y_max/15, text_label_background, fontsize=16, color='gray')

            adjustment_idx += 1

        if not cutpoints_label_bool:
            plt.axvline(region_end, ls="--", color="black", alpha=0.5)

        return adjustment_idx

    def real_adjustment_factors(self, instances, cutpoints) -> torch.tensor:
        """
        Calculate the real adjustment factors.
        """
        instances = self._to_tensor(instances)
        cutpoints = self._to_tensor(cutpoints)

        cdfs_baseline = self.glm.distributions(instances).cdf(cutpoints.unsqueeze(-1))
        cdfs_drn = self.drn.distributions(instances).cdf(cutpoints.unsqueeze(-1))

        prob_masses = torch.clamp(torch.diff(cdfs_drn, axis=0), min=1e-12)
        baseline_probs = torch.clamp(torch.diff(cdfs_baseline, axis=0), min=1e-12)
        epsilon = 1e-15
        real_adjustment_factors = (prob_masses + epsilon) / (baseline_probs + epsilon)

        return torch.clamp(real_adjustment_factors, min=0, max=10000)

    def _to_tensor(self, data):
        """
        Convert data to a torch.Tensor if not already.
        """
        if isinstance(data, torch.Tensor):
            return data
        return torch.tensor(
            np.asarray(data),
            dtype=next(self.glm.parameters()).dtype,
            device=next(self.glm.parameters()).device,
        )

    def one_hot_encoder(self, instances):
        """
        Apply one-hot encoding to categorical features, considering all possible categories from self.all_categories
        """
        # if self.column_transformer is not None:
        #     self.column_transformer.fit_transform(self.background_data_df_before_one_hot)
        #     instances = (self.column_transformer.transform(pd.DataFrame(instances,\
        #                      columns = self.background_data_df_before_one_hot.columns)))

        instances = pd.DataFrame(
            instances, columns=self.background_data_df_before_one_hot.columns
        )

        if self.cat_features:
            for feature in self.cat_features:
                instances[feature] = pd.Categorical(
                    instances[feature], categories=self.all_categories[feature]
                )

            instances = pd.get_dummies(instances, columns=self.cat_features).astype(
                float
            )

        if self.column_transformer is not None:
            background_df_one_hot = pd.get_dummies(
                self.background_data_df_before_one_hot, columns=self.cat_features
            ).astype(float)
            self.column_transformer.fit_transform(background_df_one_hot)
            instances = self.column_transformer.transform(instances)
            return instances

        else:
            return instances.values
        #     instances = pd.DataFrame(instances, columns = background_df_one_hot.columns)
        # else:
        #     instances = pd.DataFrame(instances, columns = self.background_data_df_before_one_hot.columns)

        # return(instances.values)

    def mean_drn(self, instances):
        """
        Calculate the mean predicted by the DRN network given the selected instances/features
        """
        instances = (
            self._to_tensor(self.one_hot_encoder(instances))
            if self.all_categories is not None
            else self._to_tensor(instances)
        )

        return self.drn.distributions(instances).mean.detach().numpy()

    def mean_glm(self, instances):
        """
        Calculate the mean predicted by the GLM given the selected instances/features
        """
        instances = (
            self._to_tensor(self.one_hot_encoder(instances))
            if self.all_categories is not None
            else self._to_tensor(instances)
        )

        return self.glm.distributions(instances).mean.detach().numpy()

    def mean_value_function(self, instances, adjustment):
        """
        Calculate the mean value function given the selected instances/features
        """
        return (
            self.mean_drn(instances) - self.mean_glm(instances)
            if adjustment
            else self.mean_drn(instances)
        )

    def quantile_drn(self, instances, percentile=[90], grid=None):
        """
        Calculate the quantile predicted by the DRN network given the selected instances/features
        """
        grid = (
            torch.linspace(
                self.default_cutpoints[0], self.default_cutpoints[-1], 3000
            ).unsqueeze(-1)
            if grid is None
            else grid
        )

        instances = (
            self._to_tensor(self.one_hot_encoder(instances))
            if self.all_categories is not None
            else self._to_tensor(instances)
        )

        return (
            self.drn.distributions(instances)
            .quantiles(percentile, l=grid[0], u=grid[-1] * 1.5)
            .detach()
            .numpy()
        )

    def quantile_glm(self, instances, percentile=[90], grid=None):
        """
        Calculate the quantile predicted by the GLM given the selected instances/features
        """
        grid = (
            torch.linspace(
                self.default_cutpoints[0], self.default_cutpoints[-1], 3000
            ).unsqueeze(-1)
            if grid is None
            else grid
        )

        instances = (
            self._to_tensor(self.one_hot_encoder(instances))
            if self.all_categories is not None
            else self._to_tensor(instances)
        )
        return (
            self.glm.quantiles(instances, percentile, l=grid[0], u=grid[-1] * 1.5)
            .detach()
            .numpy()
            .reshape(
                instances.shape[0],
            )
        )

    def quantile_value_function(
        self, instances, adjustment, grid=None, percentile=[90]
    ):
        """
        Calculate the quantile value function given the selected instances/features
        """
        grid = (
            torch.linspace(
                self.default_cutpoints[0], self.default_cutpoints[-1], 3000
            ).unsqueeze(-1)
            if grid is None
            else grid
        )

        return (
            self.quantile_drn(instances, percentile, grid).reshape(
                instances.shape[0],
            )
            - self.quantile_glm(instances, percentile, grid).reshape(
                instances.shape[0],
            )
            if adjustment
            else self.quantile_drn(instances, percentile, grid).reshape(
                instances.shape[0],
            )
        )

    def variance_glm(self, instances):

        raise NotImplementedError(
            f"Variance calculation for the GLM yet to be implemented!"
        )

    def variance_drn(self, instances):

        raise NotImplementedError(
            f"Variance calculation for the DRN network yet to be implemented!"
        )

    def variance_value_function(self, instances, adjustment):

        raise NotImplementedError(
            f"Kernel SHAP for variance has yet to be implemented!"
        )

    def set_value_function(self, distributional_property, adjustment, model_function):
        """
        Calculate the numeric part from the distributional property XX% quantile.
        Set the value function accordingly.
        """
        match = re.search(r"(\d+)% Quantile", distributional_property)
        if match:
            # Extract the percentage as an integer
            percentile = int(match.group(1))
            value_function = lambda instances: model_function(
                instances, adjustment=adjustment, percentile=[percentile]
            )
        else:
            raise ValueError("Invalid distributional property! What Are We Explaining?")

        return value_function
