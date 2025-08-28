#!/bin/bash

# Package HTML results for easy transfer
echo "Packaging HTML results..."

# Create a tar archive of all HTML files
tar -czf results_package.tar.gz results/*.html

echo "✓ Created results_package.tar.gz"
echo ""
echo "To transfer to your local machine, run this from your Mac:"
echo "  scp root@<runpod-ip>:/root/bad_reason_vector/bad_vec/mvp/lora_mvp/results_package.tar.gz ~/Desktop/scratch/"
echo ""
echo "Then extract with:"
echo "  cd ~/Desktop/scratch && tar -xzf results_package.tar.gz"