#!/bin/bash

# build-and-push.sh
# Helper script to build and push Docker image for GKE deployment

set -e

# Configuration
IMAGE_NAME="us-central1-docker.pkg.dev/lab-test-project-1-305710/legal-processing/topolegal"
IMAGE_TAG="${1:-1}"  # Default to tag "1" if not provided
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

echo "🐳 Building Docker image for GKE deployment..."
echo "Image: ${FULL_IMAGE}"
echo "Platform: linux/amd64 (required for GKE)"
echo ""

# Check if Docker buildx is available
if ! docker buildx version > /dev/null 2>&1; then
    echo "❌ Error: Docker buildx is required but not available"
    echo "Please install Docker Desktop or enable buildx"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found in root directory"
    echo "The container will rely on Kubernetes environment variables"
fi

# Build the image for the correct architecture
echo "📦 Building image..."
docker buildx build \
    --platform linux/amd64 \
    --tag "${FULL_IMAGE}" \
    --load \
    .

echo "✅ Image built successfully!"
echo ""

# Test the image locally
echo "🧪 Testing image locally..."
if docker run --rm "${FULL_IMAGE}" --help > /dev/null 2>&1; then
    echo "✅ Basic functionality test passed"
else
    echo "❌ Basic functionality test failed"
    exit 1
fi

# Test process_batch help
if docker run --rm "${FULL_IMAGE}" process_batch --help > /dev/null 2>&1; then
    echo "✅ Process batch help test passed"
else
    echo "❌ Process batch help test failed"
    exit 1
fi

echo ""

# Check if gcloud is configured for authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "⚠️  Warning: No active gcloud authentication found"
    echo "Please run: gcloud auth login"
    echo "Or: gcloud auth configure-docker us-central1-docker.pkg.dev"
    echo ""
    echo "🔄 Skipping push step. Run with authentication to push to registry."
    exit 0
fi

# Push the image
echo "🚀 Pushing image to Google Container Registry..."
if docker push "${FULL_IMAGE}"; then
    echo "✅ Image pushed successfully!"
else
    echo "❌ Failed to push image"
    exit 1
fi

echo ""
echo "📋 Image ready for deployment: ${FULL_IMAGE}"
echo ""
echo "🚀 Next steps:"
echo "1. Ensure your Kubernetes secret is created:"
echo "   cd ci && ./setup-k8s-secret.sh"
echo ""
echo "2. Deploy to GKE:"
echo "   kubectl apply -f process-batch-cronjob.yaml"
echo ""
echo "3. Monitor deployment:"
echo "   kubectl get cronjobs"
echo "   kubectl get pods -l app=topolegal"
echo ""
echo "4. Test locally (optional):"
echo "   cd ci && ./test-gke-deployment.sh" 