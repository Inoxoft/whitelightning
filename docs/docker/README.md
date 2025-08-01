1.  **Prerequisites:**

    - Docker installed and running.
    - Docker Compose installed.
    - NVIDIA drivers and NVIDIA Container Toolkit (if you plan to use GPU and uncommented the GPU sections).

2.  **Navigate to your Project Root:**
    Open a terminal and `cd` into the directory containing your `Dockerfile`, `docker compose.yml`, etc.

3.  **Populate `.env` file:**
    Make sure your `.env` file has your `OPEN_ROUTER_API_KEY`.

4.  **Build the Docker Image:**

    ```bash
    docker build -t whitelightning:local .
    ```

5.  **Run the Classifier Generation Agent:**
    Use `docker run` to execute your agent script with the desired arguments.
6.  **General Syntax:**
    `docker run --rm -v $(pwd):/app/models -e OPEN_ROUTER_API_KEY=YOUR_KEY_HERE whitelightning:local [Your arguments]`

    **API Key Options:**

    - **Recommended:** Pass the key as an environment variable using `-e OPEN_ROUTER_API_KEY=YOUR_KEY_HERE` (overrides .env).
    - **Alternative (dev/local):** Use the .env file if you don't want to pass the key each time.

    **Examples:**

    - **Generate a binary TensorFlow classifier for "sentiment analysis":**
      ```bash
      docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/.env:/app/.env \
          -e OPEN_ROUTER_API_KEY=YOUR_KEY_HERE \
          whitelightning:local \
          -p="Classify customer feedback as positive or negative sentiment" \
          --model-type="tensorflow" \
          --refinement-cycles=1 \
          --generate-edge-cases="true" \
          --lang="english"
      ```
    - **Generate a multiclass PyTorch classifier for "topic modeling":**
      ```bash
      docker run --rm -v $(pwd)/models:/app/models -v $(pwd)/.env:/app/.env \
          -e OPEN_ROUTER_API_KEY=YOUR_KEY_HERE \
          whitelightning:local \
          -p "Categorize news articles into topics like sports, politics, or technology" \
          --model-type="pytorch" \
          # Add other relevant flags
      ```

6**Check Outputs:**
After the command finishes, check your local `models` directory on your host machine. You should find the model artifacts, ONNX file, config JSON, etc.

This comprehensive setup should allow you to build and run your text classifier agent in a containerized and reproducible environment. Remember to adjust paths and configurations in `settings.py` regarding `DEFAULT_OUTPUT_PATH` if you want it to default to `/app/models` when running inside Docker.
