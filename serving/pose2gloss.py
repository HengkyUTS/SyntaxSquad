import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from clearml import Task, PipelineController
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from .schemas import LandmarkRequest, PredictionResponse, GlossPrediction
from ..pose2gloss.model_utils import *


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL, LABEL_ENCODER, MAX_FRAMES, PAD_VALUE, NUM_LANDMARKS, MAX_LABELS
    ml_pipeline = PipelineController.get(
        pipeline_project='SyntaxSquad', 
        pipeline_name='SyntaxSquad ML Pipeline', 
        pipeline_tags=['production']
    )
    ml_pipeline_task = Task.get_task(task_id=ml_pipeline.id)
    ml_pipeline_nodes = ml_pipeline.get_processed_nodes()

    # Load hyperparameters from the pipeline
    hyperparameters = ml_pipeline_task.get_parameters()
    MAX_LABELS = int(hyperparameters['Args/max_labels'])
    MAX_FRAMES = int(hyperparameters['Args/max_frames'])
    PAD_VALUE = int(hyperparameters['Args/pad_value'])

    # Load the label encoder from the data transformation task
    data_transformation_task = Task.get_task(task_id=ml_pipeline_nodes['step3_data_transformation'].executed)
    X_test = data_transformation_task.artifacts['X_test'].get()
    LABEL_ENCODER = data_transformation_task.artifacts['label_encoder'].get()
    NUM_LANDMARKS = X_test.shape[-2]

    # Load the model weights from the best training task
    hyperparameter_tuning_task = Task.get_task(task_id=ml_pipeline_nodes['step6_hyperparameter_tuning'].executed)
    best_job_id = hyperparameter_tuning_task.get_parameter('General/best_job_id')
    best_model_training_task = Task.get_task(task_id=best_job_id)
    MODEL = load_model(best_model_training_task.models['output'][-1].get_local_copy())
    yield

    # Clean up and release the resources
    del LABEL_ENCODER
    del MODEL


app = FastAPI(
    title='Pose-to-Gloss Model Serving', lifespan=lifespan,
    description='API for predicting glosses from a list of frame landmarks using a best trained Pose-to-Gloss model from ClearML.',
)

@app.get('/health')
async def health_check(): # Health check endpoint to verify API status
    if MODEL is None or LABEL_ENCODER is None:
        raise HTTPException(status_code=503, detail='Model or label encoder not loaded')
    return {'status': 'healthy', 'model_loaded': MODEL is not None}

@app.get('/metadata')
async def get_metadata(): # Retrieve metadata about the loaded model
    metadata = {
        'max_frames': MAX_FRAMES,
        'pad_value': PAD_VALUE,
        'num_landmarks': NUM_LANDMARKS,
        'max_labels': MAX_LABELS,
        'label_encoder_classes': LABEL_ENCODER.classes_.tolist(),
        'label_encoder_mapping': {i: gloss for i, gloss in enumerate(LABEL_ENCODER.classes_)}
    }
    return JSONResponse(content=metadata)

@app.post('/predict', response_model=PredictionResponse)
async def predict(request: LandmarkRequest):
    try: # Predict glosses from frame landmarks and return top N with softmax scores
        landmarks = np.array(request.landmarks, dtype=np.float32)
        if len(landmarks.shape) != 3 or landmarks.shape[2] != 3:
            raise HTTPException(status_code=400, detail='Landmarks must have shape (frames, landmarks, 3)')
        if request.top_n < 1 or request.top_n > MAX_LABELS:
            raise HTTPException(status_code=400, detail=f'top_n must be between 1 and {MAX_LABELS}')

        # Pad or truncate to match training MAX_FRAMES
        landmarks = landmarks[:MAX_FRAMES] if len(landmarks) > MAX_FRAMES else np.pad(landmarks, (
            (0, MAX_FRAMES - len(landmarks)), # 0 for no padding before existing frames, the other for padding needed to reach MAX_FRAMES
            (0, 0), # Apply no padding to the 2nd dimension (landmarks)
            (0, 0)  # Apply no padding to the 3rd dimension (coordinates)
        ), mode='constant', constant_values=PAD_VALUE)

        # Normalize using nose center and prepare for model input
        nose_center = landmarks[:, 49, :]
        landmarks = landmarks - nose_center[:, None, :]
        landmarks = tf.expand_dims(landmarks, axis=0) # Add batch dimension : (1, MAX_FRAMES, 180, 3)

        # Run inference
        logits = MODEL(landmarks, training=False)
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()[0] # Shape: (MAX_LABELS,)
        
        # Get top N predictions and scores
        top_n_indices = np.argsort(probabilities)[-request.top_n:][::-1]
        top_n_scores = probabilities[top_n_indices]
        top_n_glosses = LABEL_ENCODER.inverse_transform(top_n_indices)

        # Format response
        predictions = [
            GlossPrediction(gloss=gloss, score=float(score))
            for gloss, score in zip(top_n_glosses, top_n_scores)
        ]
        return PredictionResponse(predictions=predictions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction error: {str(e)}')