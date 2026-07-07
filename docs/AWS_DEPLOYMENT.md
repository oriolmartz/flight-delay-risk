# AWS Deployment Guide (v1)

A practical, minimal-cost path to deploy the FlightRisk FastAPI service to
AWS. This is intentionally a "v1" path -- solid enough for a portfolio demo
or small internal tool, not a fully hardened production architecture.

## Overview

```text
Docker build (local) → push image to ECR → run container on ECS Fargate
(or a single EC2 instance) → expose via a public endpoint / load balancer
```

## 1. Build the Docker image

```bash
docker build -t flightrisk-api:latest .
```

Test it locally first:

```bash
docker run -p 8000:8000 -v $(pwd)/models:/app/models flightrisk-api:latest
curl http://localhost:8000/health
```

## 2. Push to Amazon ECR

```bash
# One-time: create the repository
aws ecr create-repository --repository-name flightrisk-api

# Authenticate Docker to ECR
aws ecr get-login-password --region <your-region> \
  | docker login --username AWS --password-stdin <account-id>.dkr.ecr.<your-region>.amazonaws.com

# Tag and push
docker tag flightrisk-api:latest <account-id>.dkr.ecr.<your-region>.amazonaws.com/flightrisk-api:latest
docker push <account-id>.dkr.ecr.<your-region>.amazonaws.com/flightrisk-api:latest
```

## 3. Deploy the container

### Option A -- ECS Fargate (recommended for a portfolio demo)

1. Create an ECS cluster (Fargate launch type).
2. Create a Task Definition:
   * Image: the ECR URI from step 2.
   * Port mapping: container port `8000`.
   * CPU/memory: `0.5 vCPU / 1 GB` is enough for this model size.
   * Environment variables: see below.
3. Create an ECS Service running 1 task (scale up later if needed), behind
   an **Application Load Balancer** with a target group health check on
   `GET /health`.
4. Open the ALB's DNS name (or attach a custom domain via Route 53 + ACM
   for HTTPS).

### Option B -- Single EC2 instance (cheapest, least resilient)

1. Launch a small instance (e.g. `t3.small`).
2. Install Docker, pull the image from ECR, and run it:
   ```bash
   docker run -d -p 8000:8000 --restart unless-stopped \
     <account-id>.dkr.ecr.<your-region>.amazonaws.com/flightrisk-api:latest
   ```
3. Open port 8000 (or 443 behind a reverse proxy / ALB) in the security
   group.

## 4. Environment variables

| Variable                 | Purpose                                             | Default |
|---------------------------|------------------------------------------------------|---------|
| `FLIGHTRISK_THRESHOLD`    | Probability threshold for binary risk decisions      | `0.5`   |

No paid third-party API keys are required for v1 -- the model artifact is
baked into the image (or mounted from a volume/S3 at startup).

**Model artifact:** for a real deployment, build the image *after* running
`scripts/train_model.py` so `models/flightrisk_model.joblib` is present and
gets copied into the image, or mount it from an S3 bucket / EFS volume at
container start.

## 5. Health check

The API exposes `GET /health`, returning `{"status": "ok", "model_loaded": true|false}`.
Use this as:

* The Docker `HEALTHCHECK` (already configured in the `Dockerfile`).
* The ALB target group health check path.
* The ECS task health check.

## 6. Cost warning

* Fargate: billed per vCPU/memory-second while running. A single small
  task (0.5 vCPU / 1 GB) left running 24/7 costs roughly **$15-20/month**
  (varies by region) -- consider scaling the service to 0 or stopping it
  when not actively demoing.
* Application Load Balancer: has its own hourly charge (~$16-20/month)
  plus data processing -- for a pure portfolio demo, a single EC2 instance
  with an Elastic IP (Option B) is cheaper if you don't need
  auto-scaling/HA.
* ECR storage and data transfer are typically negligible for an image this
  size.
* **Always set a budget alert** in AWS Billing if experimenting with this
  for the first time.

## 7. Future improvements

* Add HTTPS via ACM + ALB (or API Gateway in front of the service).
* Store the model artifact in S3 and load it at container startup instead
  of baking it into the image, so retraining doesn't require a rebuild.
* Add autoscaling policies based on CPU/request count.
* Add request logging / observability (CloudWatch Logs + a dashboard).
* Add authentication (API key or IAM-based) if the API is exposed beyond a
  portfolio demo.
* Set up CI/CD (e.g. GitHub Actions) to build, test, and push the image on
  every merge to `main`.
