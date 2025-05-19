import os
from clearml import Task, PipelineController
from pipeline_reports import get_pipeline_of_current_commit


def compare_and_tag_task(commit_hash): # Compare current performance to best previous performance and only allow equal or better
    current_pipeline = get_pipeline_of_current_commit(commit_hash)
    best_pipeline = PipelineController.get(
        pipeline_project='SyntaxSquad', 
        pipeline_name='SyntaxSquad ML Pipeline', 
        pipeline_tags=['production']
    )
    if not best_pipeline: 
        current_pipeline.add_tags(['production'])
        return
    
    current_pipeline_nodes = current_pipeline.get_processed_nodes()
    current_model_evaluation_task = Task.get_task(task_id=current_pipeline_nodes['step7_model_evaluation'].executed)
    current_test_accuracy = current_model_evaluation_task.get_reported_single_value('test_accuracy')

    best_pipeline_nodes = best_pipeline.get_processed_nodes()
    best_model_evaluation_task = Task.get_task(task_id=best_pipeline_nodes['step7_model_evaluation'].executed)
    best_test_accuracy = best_model_evaluation_task.get_reported_single_value('test_accuracy')
    
    print(f'Best test_accuracy in the system is: {best_test_accuracy} and current metric is {current_test_accuracy}')
    if current_test_accuracy >= best_test_accuracy:
        print('This means current test_accuracy is better or equal! Tagging as such.')
        best_pipeline.set_tags(list(set(best_pipeline.get_tags()) - {'production'}))
        current_pipeline.add_tags(['production'])
    else: print('This means current test_accuracy is worse! Not tagging.')


if __name__ == '__main__':
    print('Running on commit hash:', os.getenv('COMMIT_ID'))
    compare_and_tag_task(os.getenv('COMMIT_ID'))