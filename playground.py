from binary_classifier.model import run_training
from binary_classifier.model_executor import run_model
from utils.benchmark import test_model_accuracy

# Define ranges and steps for parameters
batch_sizes = [32, 64, 128]
epochs_list = [10, 20, 30, 40, 50]
embed_dims = [16, 32, 64]
hidden_dims = [32, 64, 128]

# Loop through all combinations of parameters
if __name__ == "__main__":
    for batch_size in batch_sizes:
        for epochs in epochs_list:
            for embed_dim in embed_dims:
                for hidden_dim in hidden_dims:
                    for i in range(5):
                        model_slug = f"btch-{batch_size}_ep-{epochs}_emd-{embed_dim}_hd-{hidden_dim}_iter-{i}"
                        run_training(
                            batch_size=batch_size,
                            epochs=epochs,
                            embed_dim=embed_dim,
                            hidden_dim=hidden_dim,
                            custom_models_path=f"playground_mdls/{model_slug}"
                        )
                        run_model(
                            custom_models_path=f"playground_mdls/{model_slug}",
                            custom_results_path=f"playground_results/{model_slug}",
                            embed_dim=embed_dim,
                            hidden_dim=hidden_dim
                        )
                        test_model_accuracy(custom_results_path=f"playground_results/{model_slug}")