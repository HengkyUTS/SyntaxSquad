import os
import json
import pandas as pd
from clearml import Task, PipelineController
from github3 import login
from tabulate import tabulate


def get_pipeline_of_current_commit(commit_id): # Find the ClearML task that correspond to the exact codebase in the commit ID
    # Get the ID and Diff of all tasks based on the current commit hash, order by newest
    tasks = Task.query_tasks(task_name='SyntaxSquad ML Pipeline', task_filter={
        'order_by': ['-last_update'], 'status': ['completed'], 
        '_all_': dict(fields=['script.version_num'], pattern=commit_id)
    }, additional_return_fields=['script.diff'])

    # If no task was run yet with the exact PR code, raise an error and block the PR.
    if not tasks: raise ValueError('No task based on this code found in ClearML. Make sure to run it at least once before merging.')
    for task in tasks: # If there are tasks, check which one has no diff: aka which one was run with the exact code staged in this PR
        if not task['script.diff']:
            return PipelineController.get(pipeline_id=task['id'])
            

def get_reported_table(task, title): # Get the table from the task's reported plots
    reported_plots = task.get_reported_plots()
    metrics_report = next((d for d in reported_plots if title.lower() in d.get('metric').lower()), None)
    if not metrics_report: return None
    metrics_report = json.loads(metrics_report['plot_str'])['data'][0]
    df = pd.DataFrame(list(zip(*metrics_report['cells']['values'])), columns=[col[0] for col in metrics_report['header']['values']])
    return tabulate(df, tablefmt='github', headers='keys', showindex=False)


def get_task_stats(task, title): # Get the comment markdown for a stats table based on the task object
    task_status = task.get_status() # Try to get the task stats
    output_log_page = task.get_output_log_web_page()

    if task_status == 'completed':
        table = get_reported_table(task, title)
        if table: return f'{title}:\n\n{table}\n\nYou can view full task results [here]({output_log_page})'
        return (f'Something went wrong when creating the task table. '
                f'Check full task [here]({output_log_page})')
    return f'Task is in {task_status} status, this should not happen!'


def create_stats_comment(task_stats): # Create a comment on the current PR containing the ClearML task stats
    payload_fname = os.getenv('GITHUB_EVENT_PATH')
    with open(payload_fname, 'r') as f:
        payload = json.load(f)
        print(payload)

    owner, repo = payload.get('repository', {}).get('full_name', '').split('/')
    if owner and repo:
        gh = login(token=os.getenv('GH_TOKEN'))
        if gh:
            pull_request = gh.pull_request(owner, repo, payload.get('number'))
            if pull_request: pull_request.create_comment(task_stats) # Add the task metrics to the PR automatically
            else: print(f"Can't comment PR, {payload.get('number')}")
        else: print(f"Can't log in to gh, {os.getenv('GH_TOKEN')}")


if __name__ == '__main__': # Main check: Does a ClearML task exist for this specific commit?
    print('Running on commit hash:', os.getenv('COMMIT_ID'))
    ml_pipeline_task = get_pipeline_of_current_commit(os.getenv('COMMIT_ID'))
    ml_pipeline_task.add_tags(['main_branch']) # If the task exists, tag it as such, so we know in the interface which one it is
    ml_pipeline_nodes = ml_pipeline_task.get_processed_nodes()
    model_evaluation_task = Task.get_task(task_id=ml_pipeline_nodes['step7_model_evaluation'].executed)
    task_stats = get_task_stats(model_evaluation_task, 'Model evaluation metrics on 3 subsets')
    create_stats_comment(task_stats) # Get the metrics from the task and create a comment on the PR