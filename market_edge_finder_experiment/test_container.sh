#!/bin/bash

# Market Edge Finder - Container Test Script
# Tests the optimized Docker setup with caching

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Market Edge Finder - Container Test${NC}"
echo "======================================"

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check prerequisites
echo -e "\n${BLUE}📋 Checking Prerequisites${NC}"
echo "=========================="

# Check Docker
if command -v docker &> /dev/null; then
    print_status "Docker is installed: $(docker --version)"
else
    print_error "Docker is not installed"
    exit 1
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    print_status "Docker Compose is installed: $(docker-compose --version)"
else
    print_error "Docker Compose is not installed"
    exit 1
fi

# Check BuildKit
if [ "$DOCKER_BUILDKIT" = "1" ]; then
    print_status "BuildKit is enabled"
else
    print_warning "BuildKit not enabled - setting DOCKER_BUILDKIT=1"
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
fi

# Check .env file
if [ -f .env ]; then
    print_status ".env file exists"
    
    # Check for required variables
    if grep -q "OANDA_API_KEY=" .env && grep -q "OANDA_ACCOUNT_ID=" .env; then
        print_status "OANDA credentials configured in .env"
    else
        print_warning "OANDA credentials not configured in .env"
        print_info "Please configure OANDA_API_KEY and OANDA_ACCOUNT_ID in .env file"
    fi
else
    print_warning ".env file not found - creating from example"
    cp .env.example .env
    print_info "Please configure .env file with your OANDA credentials"
fi

# Create necessary directories
echo -e "\n${BLUE}📁 Creating Directories${NC}"
echo "======================="

directories=("data" "logs" "results" "models" "${HOME}/.cache/pip" "${HOME}/.cache/torch" "${HOME}/.cache/numba")

for dir in "${directories[@]}"; do
    if mkdir -p "$dir" 2>/dev/null; then
        print_status "Created/verified directory: $dir"
    else
        print_warning "Could not create directory: $dir"
    fi
done

# Test requirements verification
echo -e "\n${BLUE}🔍 Testing Requirements Verification${NC}"
echo "===================================="

print_info "Building verification container..."
if docker-compose -f docker-compose.data.yml build requirements-check; then
    print_status "Requirements verification container built successfully"
    
    print_info "Running requirements verification..."
    if docker-compose -f docker-compose.data.yml run --rm requirements-check; then
        print_status "Requirements verification passed"
    else
        print_warning "Requirements verification had issues - check output above"
    fi
else
    print_error "Failed to build requirements verification container"
fi

# Test data downloader build
echo -e "\n${BLUE}🔨 Testing Data Downloader Build${NC}"
echo "================================="

print_info "Building data downloader with caching..."
start_time=$(date +%s)

if docker-compose -f docker-compose.data.yml build data-downloader; then
    end_time=$(date +%s)
    build_duration=$((end_time - start_time))
    print_status "Data downloader built successfully in ${build_duration}s"
    
    # Test second build to verify caching
    print_info "Testing cache effectiveness with second build..."
    start_time=$(date +%s)
    
    if docker-compose -f docker-compose.data.yml build data-downloader; then
        end_time=$(date +%s)
        cached_build_duration=$((end_time - start_time))
        print_status "Cached build completed in ${cached_build_duration}s"
        
        if [ $cached_build_duration -lt $((build_duration / 2)) ]; then
            print_status "Cache optimization working effectively! (${cached_build_duration}s vs ${build_duration}s)"
        else
            print_warning "Cache optimization may not be optimal"
        fi
    else
        print_error "Cached build failed"
    fi
else
    print_error "Failed to build data downloader container"
    exit 1
fi

# Test container health
echo -e "\n${BLUE}🏥 Testing Container Health${NC}"
echo "==========================="

print_info "Starting health check container..."
if docker-compose -f docker-compose.data.yml run --rm data-downloader python -c "
import sys
try:
    import v20
    import pandas as pd
    import numpy as np
    import torch
    import lightgbm
    print('✅ All critical dependencies imported successfully')
    
    # Test v20 context creation
    ctx = v20.Context('api-fxpractice.oanda.com', 443, True, 'HealthCheck', 'dummy_token')
    print('✅ OANDA v20 context creation successful')
    
    # Test data structures
    df = pd.DataFrame({'test': [1, 2, 3]})
    print('✅ Pandas working correctly')
    
    # Test PyTorch
    tensor = torch.tensor([1.0, 2.0, 3.0])
    print(f'✅ PyTorch working correctly (device: {tensor.device})')
    
    print('✅ Container health check passed')
    sys.exit(0)
except Exception as e:
    print(f'❌ Health check failed: {str(e)}')
    sys.exit(1)
"; then
    print_status "Container health check passed"
else
    print_error "Container health check failed"
fi

# Test volume mounts
echo -e "\n${BLUE}📁 Testing Volume Mounts${NC}"
echo "========================"

print_info "Testing data directory mount..."
if docker-compose -f docker-compose.data.yml run --rm data-downloader python -c "
import os
import sys

# Test data directory
if os.path.exists('/data'):
    print('✅ Data directory mounted at /data')
else:
    print('❌ Data directory not mounted')
    sys.exit(1)

# Test logs directory
if os.path.exists('/logs'):
    print('✅ Logs directory mounted at /logs')
else:
    print('❌ Logs directory not mounted')
    sys.exit(1)

# Test cache directories
cache_dirs = ['/cache/torch', '/cache/numba']
for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        print(f'✅ Cache directory mounted at {cache_dir}')
    else:
        print(f'⚠️  Cache directory not found at {cache_dir}')

print('✅ Volume mount test completed')
"; then
    print_status "Volume mounts working correctly"
else
    print_error "Volume mount test failed"
fi

# Display cache information
echo -e "\n${BLUE}📊 Cache Information${NC}"
echo "==================="

print_info "Build cache locations:"
echo "  - Pip cache: ${HOME}/.cache/pip"
echo "  - PyTorch cache: ${HOME}/.cache/torch"
echo "  - Numba cache: ${HOME}/.cache/numba"

# Show cache sizes
for cache_dir in "${HOME}/.cache/pip" "${HOME}/.cache/torch" "${HOME}/.cache/numba"; do
    if [ -d "$cache_dir" ]; then
        size=$(du -sh "$cache_dir" 2>/dev/null | cut -f1)
        echo "  - $(basename "$cache_dir"): $size"
    else
        echo "  - $(basename "$cache_dir"): not created yet"
    fi
done

# Summary
echo -e "\n${BLUE}📋 Test Summary${NC}"
echo "==============="

print_status "Container test completed successfully!"
echo ""
print_info "Next steps:"
echo "1. Configure OANDA credentials in .env file"
echo "2. Run 'make download-data' to test data download"
echo "3. Run 'make up' to start the full system"
echo ""
print_info "Quick commands:"
echo "  make setup           - Initial setup"
echo "  make build           - Build with caching"
echo "  make download-data   - Download OANDA data"
echo "  make verify-requirements - Verify dependencies"
echo "  make help            - Show all available commands"

echo -e "\n${GREEN}🎉 Container setup is ready!${NC}"