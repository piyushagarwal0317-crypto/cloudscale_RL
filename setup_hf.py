#!/usr/bin/env python3
"""
HF Spaces Setup Script for CloudScale RL
Run this script to authenticate and deploy your environment to Hugging Face Spaces.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("🚀 Setting up CloudScale RL for Hugging Face Spaces")
    print()

    # Check if HF_TOKEN is set
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN environment variable is not set!")
        print()
        print("Please set your Hugging Face token by running:")
        print("export HF_TOKEN='your_token_here'")
        print()
        print("You can get your token from: https://huggingface.co/settings/tokens")
        print()
        print("Or run this script with: HF_TOKEN=your_token python setup_hf.py")
        sys.exit(1)

    print("✅ HF_TOKEN is set")

    # Authenticate with Hugging Face
    print("🔐 Authenticating with Hugging Face...")
    try:
        from huggingface_hub import login
        login(hf_token)
        print("✅ Authentication successful!")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)

    print()
    print("📦 Pushing to Hugging Face Spaces...")

    # Get repo name from git remote or use default
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, check=True
        )
        repo_url = result.stdout.strip()
        repo_name = Path(repo_url).stem  # Remove .git extension
    except:
        repo_name = "cloudscale-rl"

    print(f"Using repository name: {repo_name}")

    # Push to HF Spaces
    repo_id = f"bitmain/{repo_name}"
    cmd = ["openenv", "push", "--repo-id", repo_id]

    try:
        result = subprocess.run(cmd, check=True)
        print()
        print("🎉 Deployment successful!")
        print(f"Your space should be available at: https://huggingface.co/spaces/{repo_id}")
    except subprocess.CalledProcessError as e:
        print()
        print("❌ Deployment failed!")
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()