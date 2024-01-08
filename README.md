# IT Artefact for CSCK700 - James Grant

## Setup

### 1. Local Container

Instructions for setting up the project on Paperspace Gradient.

#### Cloning the Repository

To clone this repository locally, follow these steps:

1. Open a terminal or command prompt.
2. Navigate to the directory where you want to clone the repository.
3. Run the following command:

    ```bash
    git clone https://github.com/jrgrant-uliv/capstone-project-csck700.git
    cd 
    ```

4. Once the cloning process is complete, you will have a local copy of the repository on your machine.

#### Pull the container image

To pull a public container image from ghcr.io, follow these steps:

1. Open a terminal or command prompt.
2. Run the following command:

    ```bash
    docker pull ghcr.io/username/repo:tag
    ```

    Replace `username` with the GitHub username or organization name, `repo` with the name of the repository, and `tag` with the desired version or tag of the container image.

3. Wait for the container image to be downloaded.

#### Run the project in the docker container

To start the docker container with the working directory mounted, use the following command:

    ```bash
     docker run --rm --runtime=nvidia --gpus all -p 8888:8888 -v $(pwd)/it_artefact/:/App/ ghcr.io/jrgrant-uliv/tensorflow-cuda-conda:v1 
    ```

### 2. Paperspace Gradient

### 3. Google Colab

Instructions for setting up the project on Google Colab.

