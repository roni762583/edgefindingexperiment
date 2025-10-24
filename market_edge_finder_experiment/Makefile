# Market Edge Finder - Docker Management Makefile
# Optimized for caching and local development

.PHONY: help build build-no-cache download-data verify-requirements clean logs status

# Enable BuildKit for advanced caching
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

# Default environment
ENV ?= development

# Colors for output
RED=\033[0;31m
GREEN=\033[0;32m
YELLOW=\033[1;33m
BLUE=\033[0;34m
NC=\033[0m # No Color

help: ## Show this help message
	@echo "${BLUE}Market Edge Finder - Docker Management${NC}"
	@echo ""
	@echo "${GREEN}Available targets:${NC}"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  ${YELLOW}%-20s${NC} %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ================================
# Build Operations with Caching
# ================================

build: ## Build all images with caching optimization
	@echo "${GREEN}🔨 Building Market Edge Finder with optimized caching...${NC}"
	@mkdir -p ${HOME}/.cache/pip ${HOME}/.cache/torch ${HOME}/.cache/numba
	docker-compose build --parallel
	@echo "${GREEN}✅ Build completed with caching optimization${NC}"

build-no-cache: ## Build all images without using cache
	@echo "${YELLOW}🔨 Building Market Edge Finder without cache...${NC}"
	docker-compose build --no-cache --parallel
	@echo "${GREEN}✅ Build completed without cache${NC}"

build-data: ## Build only data downloader image
	@echo "${GREEN}🔨 Building data downloader with caching...${NC}"
	@mkdir -p ${HOME}/.cache/pip ${HOME}/.cache/torch ${HOME}/.cache/numba
	docker-compose -f docker-compose.data.yml build
	@echo "${GREEN}✅ Data downloader built${NC}"

# ================================
# Data Download Operations
# ================================

download-data: ## Download OANDA historical data (with mounted local directory)
	@echo "${GREEN}📊 Starting OANDA data download...${NC}"
	@if [ ! -f .env ]; then echo "${RED}❌ .env file not found. Copy .env.example to .env and configure${NC}"; exit 1; fi
	@mkdir -p data logs results
	docker-compose -f docker-compose.data.yml up data-downloader
	@echo "${GREEN}✅ Data download completed. Check ./data directory${NC}"

download-data-detached: ## Download data in background
	@echo "${GREEN}📊 Starting OANDA data download in background...${NC}"
	@mkdir -p data logs results
	docker-compose -f docker-compose.data.yml up -d data-downloader
	@echo "${GREEN}✅ Data download started in background${NC}"

# ================================
# Verification and Testing
# ================================

verify-requirements: ## Verify all requirements and dependencies
	@echo "${GREEN}🔍 Verifying requirements...${NC}"
	docker-compose -f docker-compose.data.yml up requirements-check --profile verification
	@echo "${GREEN}✅ Requirements verification completed${NC}"

test: ## Run test suite in container
	@echo "${GREEN}🧪 Running tests...${NC}"
	docker-compose run --rm training-service python -m pytest tests/ -v
	@echo "${GREEN}✅ Tests completed${NC}"

# ================================
# Development Operations
# ================================

dev: ## Start development environment with Jupyter
	@echo "${GREEN}🔧 Starting development environment...${NC}"
	docker-compose --profile dev up -d dev-service
	@echo "${GREEN}✅ Development environment running at http://localhost:8888${NC}"

shell: ## Get shell access to development container
	@echo "${GREEN}🐚 Opening development shell...${NC}"
	docker-compose run --rm dev-service bash

jupyter: ## Start Jupyter Lab in development container
	@echo "${GREEN}📓 Starting Jupyter Lab...${NC}"
	docker-compose --profile dev up dev-service

# ================================
# Production Operations
# ================================

up: ## Start all production services
	@echo "${GREEN}🚀 Starting production services...${NC}"
	docker-compose up -d
	@echo "${GREEN}✅ All services started${NC}"

down: ## Stop all services
	@echo "${YELLOW}⏹️  Stopping all services...${NC}"
	docker-compose down
	@echo "${GREEN}✅ All services stopped${NC}"

restart: ## Restart all services
	@echo "${YELLOW}🔄 Restarting services...${NC}"
	docker-compose restart
	@echo "${GREEN}✅ Services restarted${NC}"

# ================================
# Monitoring and Maintenance
# ================================

status: ## Show status of all services
	@echo "${GREEN}📊 Service Status:${NC}"
	docker-compose ps

logs: ## Show logs from all services
	@echo "${GREEN}📋 Service Logs:${NC}"
	docker-compose logs -f

logs-data: ## Show logs from data downloader
	@echo "${GREEN}📋 Data Downloader Logs:${NC}"
	docker-compose -f docker-compose.data.yml logs -f data-downloader

health: ## Check health of all services
	@echo "${GREEN}🏥 Health Check:${NC}"
	docker-compose exec training-service python scripts/verify_requirements.py

# ================================
# Cleanup Operations
# ================================

clean: ## Clean up unused Docker resources
	@echo "${YELLOW}🧹 Cleaning up Docker resources...${NC}"
	docker system prune -f
	docker volume prune -f
	@echo "${GREEN}✅ Cleanup completed${NC}"

clean-all: ## Clean everything including volumes and images
	@echo "${RED}🗑️  WARNING: This will remove all containers, images, and volumes${NC}"
	@echo "${RED}Press Ctrl+C to cancel or Enter to continue...${NC}"
	@read confirm
	docker-compose down -v --rmi all
	docker system prune -af --volumes
	@echo "${GREEN}✅ Complete cleanup finished${NC}"

clean-cache: ## Clear build cache directories
	@echo "${YELLOW}🧹 Clearing build cache...${NC}"
	rm -rf ${HOME}/.cache/pip/wheels
	rm -rf ${HOME}/.cache/torch
	rm -rf ${HOME}/.cache/numba
	@echo "${GREEN}✅ Cache cleared${NC}"

# ================================
# Quick Start Commands
# ================================

quickstart: build download-data verify-requirements ## Complete setup and data download
	@echo "${GREEN}🎉 Quickstart completed!${NC}"
	@echo "${BLUE}Next steps:${NC}"
	@echo "  1. Configure .env file with your OANDA credentials"
	@echo "  2. Run 'make up' to start training"
	@echo "  3. Run 'make logs' to monitor progress"

setup: ## Initial setup - create directories and copy example files
	@echo "${GREEN}🔧 Setting up Market Edge Finder...${NC}"
	@mkdir -p data logs results models configs
	@if [ ! -f .env ]; then cp .env.example .env; echo "${YELLOW}📝 Created .env file - please configure your OANDA credentials${NC}"; fi
	@echo "${GREEN}✅ Setup completed${NC}"

# ================================
# Information Commands
# ================================

info: ## Show system information
	@echo "${BLUE}Market Edge Finder - System Information${NC}"
	@echo "${GREEN}Docker Version:${NC}"
	@docker --version
	@echo "${GREEN}Docker Compose Version:${NC}"
	@docker-compose --version
	@echo "${GREEN}BuildKit Status:${NC}"
	@echo "DOCKER_BUILDKIT=${DOCKER_BUILDKIT}"
	@echo "${GREEN}Available Disk Space:${NC}"
	@df -h . | tail -1
	@echo "${GREEN}Cache Directory Size:${NC}"
	@du -sh ${HOME}/.cache 2>/dev/null || echo "Cache directory not found"