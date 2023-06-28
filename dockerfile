# Base image
FROM python:3.10.2

# Set the working directory
WORKDIR /home/image_retrieval


# # Copy the requirements file
COPY requirements.txt .

RUN apt-get update && apt-get install -y libgl1-mesa-glx


# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install opencv-python
RUN python -m spacy download en_core_web_sm

COPY . .

# Expose the port
EXPOSE 5000

# Run the command to start the application
<<<<<<< HEAD
# CMD ["python", "app.py"]
CMD ["python", "app.py", "--host", "0.0.0.0"]
=======
CMD ["python", "app.py"]

>>>>>>> 5e3efbc90aefdc675cad00d380c21513a0d38261

# # Copy the requirements file
# COPY requirements.txt .

# # Install the dependencies
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the project files to the working directory
# COPY . .

# # Expose the port
# EXPOSE 5000

# # Run the command to start the application
# CMD ["python", "app.py"]
