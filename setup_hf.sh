#!/bin/bash
# HF Spaces Setup Script for CloudScale RL

echo "🚀 Setting up CloudScale RL for Hugging Face Spaces"
echo ""

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN environment variable is not set!"
    echo ""
    echo "Please set your Hugging Face token:"
    echo "export HF_TOKEN='your_token_here'"
    echo ""
    echo "You can get your token from: https://huggingface.co/settings/tokens"
    echo ""
    echo "Or run this script with: HF_TOKEN=your_token ./setup_hf.sh"
    exit 1
fi

echo "✅ HF_TOKEN is set"

# Authenticate with Hugging Face
echo "🔐 Authenticating with Hugging Face..."
python -c "from huggingface_hub import login; login('$HF_TOKEN')"

if [ $? -eq 0 ]; then
    echo "✅ Authentication successful!"
else
    echo "❌ Authentication failed!"
    exit 1
fi

echo ""
echo "📦 Pushing to Hugging Face Spaces..."

# Use the repo name from the git remote or default
REPO_NAME=$(basename $(git remote get-url origin 2>/dev/null | sed 's/.git$//') 2>/dev/null || echo "cloudscale-rl")

echo "Using repository name: $REPO_NAME"

# Push to HF Spaces
openenv push --repo-id "bitmain/$REPO_NAME"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Deployment successful!"
    echo "Your space should be available at: https://huggingface.co/spaces/bitmain/$REPO_NAME"
else
    echo ""
    echo "❌ Deployment failed!"
    exit 1
fi