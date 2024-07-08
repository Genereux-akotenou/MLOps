1. Build network for apps interc communication
```bash
docker network create ml-stack2-network --driver bridge
```

2. Run docker
```bash
docker compose up -d
```