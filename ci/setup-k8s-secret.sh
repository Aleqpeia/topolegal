#!/bin/bash

# setup-k8s-secret.sh
# Helper script to create Kubernetes secret for Google Cloud credentials

set -e

echo "🔐 Setting up Google Cloud credentials secret for GKE deployment..."

# Check if legal_key.json exists
if [ ! -f "../legal_key.json" ]; then
    echo "❌ Error: legal_key.json not found in the root directory"
    echo "Please ensure your Google Cloud service account key is available as legal_key.json"
    exit 1
fi

# Create the secret with the correct structure for GKE
echo "📝 Creating Kubernetes secret 'google-cloud-key'..."
kubectl create secret generic google-cloud-key \
    --from-file=key.json=../legal_key.json \
    --dry-run=client -o yaml | kubectl apply -f -

echo "✅ Secret 'google-cloud-key' created successfully!"
echo ""
echo "📋 Next steps for GKE deployment:"
echo "1. Build the Docker image for the correct architecture:"
echo "   docker buildx build --platform linux/amd64 -t us-central1-docker.pkg.dev/lab-test-project-1-305710/legal-processing/topolegal:1 ."
echo ""
echo "2. Push the image to Google Container Registry:"
echo "   docker push us-central1-docker.pkg.dev/lab-test-project-1-305710/legal-processing/topolegal:1"
echo ""
echo "3. Deploy the CronJob to GKE:"
echo "   kubectl apply -f ../process-batch-cronjob.yaml"
echo ""
echo "🔍 To monitor the CronJob in GKE:"
echo "   kubectl get cronjobs"
echo "   kubectl get jobs -l app=topolegal"
echo "   kubectl logs -l app=topolegal,component=batch-processor"
echo ""
echo "💡 Pro tip: Use GKE Workload Identity for better security:"
echo "   https://cloud.google.com/kubernetes-engine/docs/how-to/workload-identity" 