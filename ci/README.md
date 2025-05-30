# Topolegal GKE CI/CD Setup

This directory contains Kubernetes configurations for running the `process_batch` command on Google Kubernetes Engine (GKE).

## 📁 Files

- `../process-batch-cronjob.yaml` - Kubernetes CronJob optimized for GKE
- `setup-k8s-secret.sh` - Helper script to create Google Cloud credentials secret
- `build-and-push.sh` - Script to build and push Docker image with correct architecture
- `README.md` - This file

## 🚀 Quick Start for GKE

### 1. Prerequisites

- GKE cluster access with `kubectl` configured
- Docker with buildx support (Docker Desktop recommended)
- Google Cloud CLI (`gcloud`) installed and authenticated
- Google Cloud service account key (`legal_key.json`) in the root directory

### 2. Build and Push Docker Image

The most important step to avoid "exec format error":

```bash
# Use the provided script (recommended)
cd ci
./build-and-push.sh

# Or manually with correct architecture
docker buildx build --platform linux/amd64 \
  -t us-central1-docker.pkg.dev/lab-test-project-1-305710/legal-processing/topolegal:1 .
docker push us-central1-docker.pkg.dev/lab-test-project-1-305710/legal-processing/topolegal:1
```

### 3. Setup Google Cloud Credentials

```bash
cd ci
./setup-k8s-secret.sh
```

### 4. Deploy to GKE

```bash
kubectl apply -f process-batch-cronjob.yaml
```

## 🐛 Troubleshooting "exec format error"

The "exec format error" typically occurs due to architecture mismatch. Here's how we fixed it:

### Root Causes:
1. **Architecture Mismatch**: Building on Apple Silicon (ARM64) for GKE (AMD64)
2. **Virtual Environment Path Issues**: Incorrect Python path in container
3. **Missing Platform Specification**: Docker building for wrong architecture

### Solutions Applied:

1. **Fixed Dockerfile**:
   ```dockerfile
   FROM --platform=linux/amd64 python:3.12-slim-bookworm AS build
   # Ensures AMD64 architecture for GKE compatibility
   ```

2. **Proper Virtual Environment Handling**:
   ```dockerfile
   ENV PATH="/app/.venv/bin:$PATH"
   COPY --from=build /app/.venv /app/.venv
   ```

3. **Security Enhancements**:
   ```dockerfile
   USER appuser  # Non-root user
   ```

4. **Build Command**:
   ```bash
   docker buildx build --platform linux/amd64 ...
   ```

## 📋 GKE-Specific Features

### Security
- **Non-root user**: Container runs as `appuser` (UID 65534)
- **Security context**: Enforced in Kubernetes manifest
- **Secret mounting**: Proper volume mounting for credentials

### Monitoring
- **Health checks**: Liveness probe for container health
- **Labels**: Proper labeling for monitoring and selection
- **Resource limits**: CPU and memory limits for efficient resource usage

### Scheduling
- **Node selector**: Targets specific node pools if needed
- **Tolerations**: Ensures scheduling on AMD64 nodes
- **Anti-affinity**: Prevents overlapping jobs with `concurrencyPolicy: Forbid`

## 🔍 Monitoring

### Check CronJob Status
```bash
kubectl get cronjobs -l app=topolegal
kubectl describe cronjob topolegal-process-batch
```

### View Job Runs
```bash
kubectl get jobs -l app=topolegal
kubectl get pods -l app=topolegal,component=batch-processor
```

### Check Logs
```bash
# Get logs from the latest job
kubectl logs -l app=topolegal,component=batch-processor --tail=100

# Follow logs in real-time
kubectl logs -l app=topolegal,component=batch-processor -f
```

### Monitor Resource Usage
```bash
kubectl top pods -l app=topolegal
```

## 🛠️ Advanced Configuration

### Using GKE Workload Identity (Recommended)

For production deployments, consider using Workload Identity instead of service account keys:

1. **Enable Workload Identity on cluster**:
   ```bash
   gcloud container clusters update CLUSTER_NAME \
     --workload-pool=PROJECT_ID.svc.id.goog
   ```

2. **Create Kubernetes Service Account**:
   ```bash
   kubectl create serviceaccount topolegal-ksa
   ```

3. **Bind to Google Service Account**:
   ```bash
   gcloud iam service-accounts add-iam-policy-binding \
     --role roles/iam.workloadIdentityUser \
     --member "serviceAccount:PROJECT_ID.svc.id.goog[default/topolegal-ksa]" \
     GSA_NAME@PROJECT_ID.iam.gserviceaccount.com
   ```

4. **Update CronJob to use Workload Identity**:
   ```yaml
   spec:
     template:
       spec:
         serviceAccountName: topolegal-ksa
         # Remove volume mounts and environment variables for credentials
   ```

### Custom Node Pools

For resource-intensive NLP processing:

```yaml
nodeSelector:
  cloud.google.com/gke-nodepool: nlp-processing-pool
```

### Scaling Considerations

- **Vertical Scaling**: Increase memory/CPU limits
- **Horizontal Scaling**: Modify schedule or use multiple CronJobs for different partitions
- **Spot Instances**: Use preemptible nodes for cost optimization

## 🔧 Customization

### Processing Specific Partitions
```yaml
args: ["--use-pretrained", "process_batch", "--batch", "123"]
```

### Changing Schedule
```yaml
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
```

### Resource Adjustment
```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"
``` 