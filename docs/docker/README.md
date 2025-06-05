1.  **Prerequisites:**
    *   Docker installed and running.
    *   Docker Compose installed.
    *   NVIDIA drivers and NVIDIA Container Toolkit (if you plan to use GPU and uncommented the GPU sections).

2.  **Navigate to your Project Root:**
    Open a terminal and `cd` into the directory containing your `Dockerfile`, `docker compose.yml`, etc.

3.  **Populate `.env` file:**
    Make sure your `.env` file has your `OPEN_ROUTER_API_KEY`.

4.  **Build the Docker Image:**
    You can build it explicitly or `docker compose run` will build it if it doesn't exist.
    ```bash
    docker compose build
    ```
    Or, if you want to build directly with Docker (less common if using compose):
    ```bash
    # docker build -t text_classifier_agent_tf .
    ```

5.  **Run the Classifier Generation Agent:**
    Use `docker compose run` to execute your agent script with the desired arguments. The `run` command starts a *new container* based on the service definition, executes the command you provide, and then typically stops/removes the container (unless it's kept running by `tail -f /dev/null` and you `exec` into it).

    **General Syntax:**
    `docker compose run [--rm] <service_name> python /app/text_classifier_agent.py [your_agent_arguments]`

    *   `--rm`: Automatically removes the container when it exits (good for CLI tasks).
    *   `classifier_agent`: This is the service name from your `docker compose.yml`.
    *   `/app/text_classifier_agent.py`: The path to your script inside the container.

    **Examples:**

    *   **Generate a binary TensorFlow classifier for "sentiment analysis":**
        ```bash
        docker compose run --rm classifier_agent python /app/text_classifier_agent.py \
            -p "Classify customer feedback as positive or negative sentiment" \
            --model-type tensorflow \
            --output-path /app/generated_classifiers \
            --refinement-cycles 1 \
            --generate-edge-cases true \
            --lang english \
            --max-features 3000
        ```
        *Notice `--output-path /app/generated_classifiers`. This path is *inside the container*, but because of the volume mount in `docker compose.yml`, the actual files will appear in the `generated_classifiers` directory on your host machine.*

    *   **Generate a multiclass PyTorch classifier for "topic modeling":**
        ```bash
        docker compose run --rm classifier_agent python /app/text_classifier_agent.py \
            -p "Categorize news articles into topics like sports, politics, or technology" \
            --model-type pytorch \
            --output-path /app/generated_classifiers \
            # Add other relevant flags
        ```

6**Check Outputs:**
    After the command finishes, check your local `generated_classifiers` directory (or whatever you named it) on your host machine. You should find the model artifacts, ONNX file, config JSON, etc.

7.  **Interactive Session (for debugging or exploration):**
    If you want to poke around inside the container or run commands step-by-step:
    ```bash
    # Start the service (it will run `tail -f /dev/null` and stay up)
    docker compose up -d classifier_agent

    # Execute a bash shell in the running container
    docker compose exec classifier_agent bash

    # Now you are inside the container at /app
    # You can run your python script directly:
    # root@container_id:/app# python text_classifier_agent.py -p "Test" ...
    # Exit the shell with `exit`
    ```
    Then stop the service:
    ```bash
    docker compose down
    ```

This comprehensive setup should allow you to build and run your text classifier agent in a containerized and reproducible environment. Remember to adjust paths and configurations in `settings.py` regarding `DEFAULT_OUTPUT_PATH` if you want it to default to `/app/generated_classifiers` when running inside Docker.
