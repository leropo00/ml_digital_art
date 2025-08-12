# Digital Art AI agents and ML training experiments

Projects with AI agents and ML training notebooks,
for the domain of digital art creation.
This is still a project in the beginning, so many changes will likely happen.

## Project setup

Currently, the project has a Docker folder with a container for the project database. Python is not inside the Docker container, so you need a local instance of Python to run the project. I use WSL for development. 
The project and all its commands are run inside a Python virtual environment. Run the following commands inside your terminal:

```
# move to the project root folder and create a virtual environment
python -m venv ./.venv

# activate the virtual environment, 
# this should also be done each time when running the project
. .venv/bin/activate

# when inside a virtual environment, you leave the virtual environment with command
deactivate

# install the dependencies, present in uv.lock
uv sync

# install this module as a package
uv pip install -e .
```

To set up your environment variables, you need to copy the `.env.example` file and rename it to `.env`. You can do this manually or use the following terminal command in:

```bash
cp .env.example .env # Linux, macOS, Git Bash, WSL
copy .env.example .env # Windows Command Prompt
```

This command creates a copy of `.env.example` and names it `.env`, allowing you to configure your environment variables specific to your setup. Change variable values to correct values, such as the actual database and keys.

Run the project dependencies by moving to the docker directory and running the command:

```
docker-compose up -d
```

When you first set up the project, database tables need to be set up.
Project uses the alembic package for managing database migrations.
To set up tables after docker container is running, 
move to the project directory and run the command:

```
alembic upgrade head
```

## Project commands

Inside the file:   `pyproject.toml`, the project's commands are defined.

When adding a new command to pyproject.toml.
Use `uv pip install -e .` so that command will be visible.

Currently, two commands are present; you can run them in any folder inside a virtual environment. 

- `server_dev`: this starts the server in development mode, with hot reloading. Go to the URL http://127.0.0.1:8000/docs , here the project swagger UI is present with all the endpoints.

- `show_examples`: when more AI agents features will be implemented, multiple examples of scenarios with filled conversations will be available here.


## Project Organization

The project structure was created from this template: 
https://github.com/datalumina/datalumina-project-template.
Some of the folders are currently empty and the structure of the project may change.

Currently, the following folders are of interest:

```
├── docker: container docker image for database 
├── notebooks: contains Jupyter Notebooks, which were tested via Google Colab.
├── project: contains the implementation of the server-side project
```
