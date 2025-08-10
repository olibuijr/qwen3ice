#!/bin/bash

# Script to push to GitHub
# You'll need to enter your GitHub username and password/token when prompted

echo "Pushing to GitHub repository..."
echo "You'll be prompted for your GitHub credentials."
echo "Note: Use a Personal Access Token instead of password for security"
echo ""

# Set the remote (already done but just to be sure)
git remote set-url origin https://github.com/olibuijr/qwen3ice.git

# Push to main branch
git push -u origin main

echo ""
echo "If successful, your code is now at: https://github.com/olibuijr/qwen3ice"
echo ""
echo "To create a Personal Access Token (if needed):"
echo "1. Go to GitHub Settings → Developer settings → Personal access tokens"
echo "2. Generate new token with 'repo' scope"
echo "3. Use the token as your password when prompted"