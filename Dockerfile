# Base image
FROM python:3.11.6

# Set working directory
WORKDIR /app



# Copy code
COPY . .
RUN apt-get update && apt-get install -y libgl1
# Install dependencies using pyproject.toml (uses pip if not poetry or hatch)
RUN pip install --upgrade pip \
 && pip install .

RUN pip install scikit-learn
RUN pip install torch torchvision

ENV USE_TORCH=1
# Default command (can override in docker-compose)
CMD ["python", "m2.py"]
