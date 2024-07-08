# The original version of this docker-file has been fork from 'Ewins518/nl2sql-ollama-llama3'

1. Run docker
```bash
docker compose up -d
```

2. Run the model locally (llama3):
```bash
docker exec -it ollama ollama run llama3
```

You can now chat with the model on the terminal
3. Execute python file on your host terminal
```bash
python request.ipynb
```

If you have GPU, go to the official  [ollama docker image](https://hub.docker.com/r/ollama/ollama) for configuration.

Enjoy!
