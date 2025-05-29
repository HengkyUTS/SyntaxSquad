# Sprint 3: Develop, Test and Deploy MLOps (MLOps level 2)

In this sprint, for the product MLOPs, I will use ClearML to:

- Develop and deploy CI/CD pipeline using Github Actions.
- Develop and deploy Hyper Parameter Tuning Component with ClearML `HyperParameterOptimizer`
- Develop and deploy [multi-model training](../pose2gloss/step4_model_training.py) and [model selection](../pose2gloss/step5_model_selection.py) Component.
- Implement the product user interface using StreamLit.

This MLOps Level 2 stage is for a rapid and reliable update of the pipelines in production. You need a robust automated continuous integration/continuous deployment (CI/CD) system that is introduced in the [MLOps level 2: CI/CD pipeline automation](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning#mlops_level_2_cicd_pipeline_automation)

Regarding the [Multi-Model Training](../pose2gloss/step4_model_training.py), I only have 1 model for now, and I don't want to develop another model from scratch at this stage but I still thought that I did need another model to fulfill this requirement. Therefore, I have an idea to simply experiment with some SOTA models on the ImageNet using Keras. I will train another model for **Pose-to-Gloss** alongside my current **GISLR** model and log each model to ClearML under separate tasks.

Specifically, I modified the script for the model training task to include the `ConvNeXtTiny` model. I have tried all the available ImageNet models in Keras, but only this one shows some convergence, though it's pretty minimal (~38% top-1 accuracy). At the end of the training, alongside publishing the best model weights to ClearML, I also reload the model with it to calculate the metrics on the validation set, which are later used for the comparison of [HPO](../pose2gloss/step6_hyperparameter_tuning.py)'s experiments.

Next, I implemented the [Model Selection](../pose2gloss/step5_model_selection.py) component by developing a ClearML task to evaluate all trained models and select the best one based on **top-1 accuracy**. Before, I planned to execute this step after finishing [HPO](../pose2gloss/step6_hyperparameter_tuning.py). This meant I had to run the [HPO](../pose2gloss/step6_hyperparameter_tuning.py) more than once to find the best parameters for each model, load the top experiments from each [HPO](../pose2gloss/step6_hyperparameter_tuning.py) task, and evaluate their performance on the validation set through the [model selection](../pose2gloss/step5_model_selection.py) task to choose the best one. However, I think that approach is ineffective because it's so time-consuming and costly, requiring multiple [HPO](../pose2gloss/step6_hyperparameter_tuning.py) runs and many Colab compute units (I don't have enough money to buy more). Therefore, I took another, more effective approach. I think this is also the correct process in real-world ML model development. Here, I will put this step before the [HPO](../pose2gloss/step6_hyperparameter_tuning.py) so that I can train 2 current models with some chosen settings first and use this step to pick the better one for further hyperparameter tuning. This allows me to run the [HPO](../pose2gloss/step6_hyperparameter_tuning.py) only once with the more potential model.

Regarding putting the [HPO](../pose2gloss/step6_hyperparameter_tuning.py) first, the in-class demo in week 10 had to execute the model training task first to obtain a base task to plug into the [HPO](../pose2gloss/step6_hyperparameter_tuning.py) steps, as it couldn't seem to pass the parameter override from [HPO](../pose2gloss/step6_hyperparameter_tuning.py) to the template training task. However, I think it's not efficient, so I did some research to find a way to directly use the template task as the base task for the [HPO](../pose2gloss/step6_hyperparameter_tuning.py), so I don't have to perform the redundant initial training. I did find a solution to do so using [ClearmlJob](https://clear.ml/docs/latest/docs/references/sdk/automation_job_clearmljob/), making my pipeline as [data transformation](../pose2gloss/step3_data_transformation.py) -> [HPO](../pose2gloss/step6_hyperparameter_tuning.py)s -> model selection -> model evaluation. Here, I can use the template task for the [HPO](../pose2gloss/step6_hyperparameter_tuning.py).

However, for now, when I developed the [multi-model training](../pose2gloss/step4_model_training.py), I didn't use the [ClearmlJob](https://clear.ml/docs/latest/docs/references/sdk/automation_job_clearmljob/) anymore as my pipeline became: [data transformation](../pose2gloss/step3_data_transformation.py) -> [multi-model training](../pose2gloss/step4_model_training.py) -> [model selection](../pose2gloss/step5_model_selection.py) -> [HPO](../pose2gloss/step6_hyperparameter_tuning.py) -> model evaluation, so I already got the base model training task for the [HPO](../pose2gloss/step6_hyperparameter_tuning.py) and no longer needed the template task. This is another advantage of my new approach, simplifying my code.

To integrate both the model training and selection tasks into the pipeline, from the [data transformation](../pose2gloss/step3_data_transformation.py) task, I split the workflow into **2 branches** for the training of 2 models. Then, I merged these branches by using them as parents for the [model selection](../pose2gloss/step5_model_selection.py). Here, I will load 2 models from 2 training tasks and evaluate them on the validation set to choose the best one for [HPO](../pose2gloss/step6_hyperparameter_tuning.py). In fact, I can chain each [step 4](../pose2gloss/step4_model_training.py) together to train each model one by one. However, that approach prevents me from taking advantage of multiple workers or agents, a feature offered by ClearML. By splitting into branches, I can start 2 Colab notebooks, use them as 2 agents with GPU accelerator, and run them in parallel in the same queue, making my training much faster.

Next, I implemented the **Hyperparameter Tuning Component**. Here, I will load the best model training task, which I have identified through the [model selection](../pose2gloss/step5_model_selection.py), and use it as the base task for the [HPO](../pose2gloss/step6_hyperparameter_tuning.py). Based on the model's name (e.g., 'GISLR'), I define a search space for various hyperparameters. This includes ranges for `batch_size`, `learning_rate`, `reduce_lr_patience`, and conditionally, `conv1d_dropout` and `last_dropout`. Then, I configure ClearML's `HyperParameterOptimizer` with `OptimizerOptuna` as the search strategy for **multi-objective tuning**, aiming to minimize `val_loss` and maximize `val_accuracy`. I also set up the number of concurrent tasks to run multiple [HPO](../pose2gloss/step6_hyperparameter_tuning.py) experiments in parallel, like in [step 4](../pose2gloss/step4_model_training.py). As the search space is quite large and may take a long time to run, I set `total_max_jobs` to **2** to launch only **2 jobs** for the optimization. This is enough to demonstrate the capability to use the [HPO](../pose2gloss/step6_hyperparameter_tuning.py).

After retrieving the top-performing experiment from the [HPO](../pose2gloss/step6_hyperparameter_tuning.py). I simply compare the metrics of this new best job with the metrics of the initial best model from the **[model selection](../pose2gloss/step5_model_selection.py)** phase. If the [HPO](../pose2gloss/step6_hyperparameter_tuning.py) finds a configuration with better validation accuracy, that job's ID, hyperparameters, and metrics are recorded as the new best. These optimal results, including the best job ID, its hyperparameters, and achieved metrics, are then printed to the console and uploaded as an artifact to the ClearML task. If no better job is found, it defaults to the best model from the initial [model selection](../pose2gloss/step5_model_selection.py).

Now, to make the final end-to-end pipeline with **7 steps** ([data splitting](../pose2gloss/step1_data_splitting.py), [data augmentation](../pose2gloss/step2_data_augmentation.py), [data transformation](../pose2gloss/step3_data_transformation.py), [multi-model training](../pose2gloss/step4_model_training.py), [model selection](../pose2gloss/step5_model_selection.py), [HPO](../pose2gloss/step6_hyperparameter_tuning.py), [model evaluation](../pose2gloss/step7_model_evaluation.py)) successfully executed, I have to use **4 machines/agents**:
- My laptop to run the pipeline itself.
- 1 remote Colab CPU machine in a queue to run steps that don't require GPU computing in the pipeline.
- 2 remote Colab A100s in another queue to run the [multi-model training](../pose2gloss/step4_model_training.py) and launch each [HPO](../pose2gloss/step6_hyperparameter_tuning.py) experiment of the best model from the [model selection](../pose2gloss/step5_model_selection.py) step in parallel.

After finishing all the above, it's time to configure GitHub Actions workflows to automatically run pipeline components on pull requests (PR) or code commits. Here, I implemented 5 distinct jobs and chained them together to form an end-to-end CI/CD workflow in `.github/workflows/pipeline.yaml`; each has a corresponding Python file in which most of the logic resides. These jobs simply start a GitHub Actions instance and run. The `pipeline.yaml` defines a CI/CD pipeline for automating testing, pipeline execution, metrics reporting, production tagging, and **FastAPI** deployment on pull requests to the main branch. All jobs use secrets for ClearML API credentials. Each job checks out the PR's head commit to ensure up-to-date code. The workflow is designed for automated, robust CI/CD with model evaluation and deployment steps:

1. **test-remote-runnable**:

- Check if the code is remotely runnable by the ClearML Agent.
- Run on pull requests to main.
- Set up Python and install ClearML.
- Launch a ClearML task (`cicd/example_task.py`) remotely and check if the task is runnable or the Agent is available using `cicd/check_remotely_runnable.py`.

Usually, it's a good idea to develop the code on local computer and only later use the ClearML Agent to remotely train the model for real. To ensure this is always possible, I automatically set up the code from the PR on a ClearML agent and listen to the output. If the Agent starts reporting iterations, it means the code is remotely runnable without issues. With this check, I can ensure that every commit on our main branch is ready for remote training. In this job, I run 1 more command apart from the accompanying `check_remotely_runnable.py`, which is the clearml-task command. I can use this command to remotely launch an existing repository. In this case, I will remotely launch the `example_task.py` file and then capture the Task ID from the console output using [ripgrep](https://github.com/BurntSushi/ripgrep). Then, I can send the Task ID to the Python script to poll its status and progress.

2. **execute-pipeline** (depends on **test-remote-runnable**):

- Configure ClearML tasks to be triggered by GitHub Actions, ensuring pipeline execution.
- Set up Python and install ClearML.
- Run the main pipeline using [pipeline_from_tasks.py](../pipeline_from_tasks.py).

For a rapid and reliable update of the pipelines in production, I need a robust automated CI/CD system that is introduced in Level 2 of MLOps.

3. **report-pipeline-metrics** (depends on **execute-pipeline**):

- Add scalars to an open PR.
- Set up Python and install dependencies.
- Run [cicd/pipeline_reports.py](../cicd/pipeline_reports.py) to comment metrics (such as results from [model evaluation](../pose2gloss/step7_model_evaluation.py) step) on the PR.

I have a particular model training experiment versioned in Git and created a new feature on a side branch. Now, when I open a PR to merge that branch, I want to ensure that this code has at least one successful task run in ClearML. To make that visible, I can then use the ClearML SDK to get the latest model metric from that specific ClearML task and automatically post it on the PR as a comment with a link to the original experiment in ClearML.

4. **production-pipeline-tagging** (depends on **execute-pipeline**):

- Set up Python and install dependencies.
- Run [cicd/production_tagging.py](../cicd/production_tagging.py) to assign the latest pipeline as production if it performs better than the current one.

This job is quite similar to the previous one, but now we want to ensure that we never merge a code change that will worsen the model's performance. So, I can again get the ClearML task corresponding to the current PR, but this time, I will compare the pipeline metrics to the ones from the previous best ClearML pipeline. I'll only allow the pipeline to succeed if the metrics are equal or better. In this way, we can guarantee the quality of our main branch.

5. **deploy-fastapi** (depends on **production-pipeline-tagging**):

- Set up Python and install serving dependencies.
- Deploy a **FastAPI** server from [serving/pose2gloss.py](../serving/pose2gloss.py), check for `health`, and notify of deployment status.
- Clean up by stopping the **FastAPI** server after completion.

This job extends the CD pipeline to deploy the **FastAPI** model serving endpoint. Ensure the deployment pulls the latest model weights from ClearML. **FastAPI** endpoint is accessible via `/health`; correctly serves predictions using the best model weights. Theoretically, GitHub Actions should deploy this API to another cloud server. However, as I'm only working with localhost in this project, I simulated that process by running **FastAPI** directly on the GitHub runner and verified if it succeeded by calling the `/health` endpoint. I started the **FastAPI** server in the background and waited 4 minutes for the server to set up and download all needed artifacts from the production pipeline.  
In order to successfully implement and pass all the above steps, I have to take many trials and errors with many commits and pull requests, which is both costly and time-consuming. The last job in this Sprint is to design a [Streamlit](https://streamlit.io/) UI layout for real-time ASL translation, which Antong has done a good job so far:

- He included a webcam feed for capturing video, a section displaying extracted landmarks, predicted glosses, and the final English translation.
- He integrated my **FastAPI** `/predict` endpoint from the last Sprint into this [Streamlit](https://streamlit.io/) app to fetch gloss predictions for each sequence position and display them with their scores in the UI.

To help the UI run smoother, I took an extra step to convert our best model into [TFLite](https://www.tensorflow.org/api_docs/python/tf/lite) format for [serving/pose2gloss.py](../serving/pose2gloss.py), enabling faster inference. Now, each prediction just takes less than 1 second to return the glosses with scores.