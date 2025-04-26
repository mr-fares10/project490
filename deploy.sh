#!/bin/bash

# Fighter Jet Position Prediction Web Service - Deployment Script

# Set colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Fighter Jet Position Prediction - Deployment ===${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Create necessary directories if they don't exist
echo -e "${GREEN}Creating required directories...${NC}"
mkdir -p web_app/uploads
mkdir -p web_app/results
mkdir -p web_app/app/models
mkdir -p web_app/app/templates
mkdir -p web_app/app/static
mkdir -p examples

# Check if advanced_processor.py exists
if [ ! -f "advanced_processor.py" ]; then
    echo -e "${RED}Error: advanced_processor.py file not found.${NC}"
    echo -e "${YELLOW}Please create the advanced_processor.py file before continuing.${NC}"
    exit 1
fi

# Check if templates exist
if [ ! -f "web_app/app/templates/enhanced.html" ]; then
    echo -e "${YELLOW}Warning: enhanced.html template not found. Creating placeholder...${NC}"
    echo "<!-- Enhanced template will be saved here -->" > web_app/app/templates/enhanced.html
fi

if [ ! -f "web_app/app/templates/simple.html" ]; then
    echo -e "${YELLOW}Warning: simple.html template not found. Creating placeholder...${NC}"
    echo "<!-- Simple template will be saved here -->" > web_app/app/templates/simple.html
fi

# Build and start the containers
echo -e "${GREEN}Building and starting Docker containers...${NC}"
docker-compose up --build -d

# Check if the containers started successfully
if [ $? -eq 0 ]; then
    echo -e "${GREEN}=== Fighter Jet Position Prediction service is now running! ===${NC}"
    echo -e "Access the web interface at ${YELLOW}http://localhost:8080${NC}"
    echo -e "To view logs: ${YELLOW}docker-compose logs -f${NC}"
    echo -e "To stop the service: ${YELLOW}docker-compose down${NC}"
else
    echo -e "${RED}Failed to start the Docker containers. Please check the error messages above.${NC}"
    exit 1
fi