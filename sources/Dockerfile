FROM continuumio/miniconda3

COPY . /app 

WORKDIR /app 
# change le pathe de travail, comme un cd

RUN conda env create --file environment.yml 
# créer un environnement conda à partir du environment.yml 

ENV PATH /opt/conda/envs/cours_lyon/bin:$PATH 
# ajoute dans le path mon conda depuis là où elle est
# le path est l'endroit où se situe les executables 

ENV CONDA_DEFAUT_ENV cours_lyon 
# définit une variable d'env 

EXPOSE 8080 
# il faut exposer un port pour communiquer avec le host 

#CMD ["python, "code-lyon/predict_synthese_sanitaire.py"]
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]

