# Use official Python image
FROM python:3.9

# Create a non-root user (as recommended by Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/

# Copy requirements.txt
COPY --chown=user requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code
COPY --chown=user . .

# Expose port (FastAPI will run on this port)
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--reload"]