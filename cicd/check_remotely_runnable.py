import sys
import time
from clearml import Task

def check_task_status(task_id, timeout=600):
    task = Task.get_task(task_id=task_id)
    start_time = time.time()
    if task:
        while time.time() - start_time < timeout:
            task_status = task.get_status()
            print(task_status)

            if task_status == 'queued': start_time = time.time() # If queued, just reset the timeout timer
            if task_status in ['failed', 'stopped']: raise ValueError('Task did not run correctly, check logs in webUI')
            elif task_status == 'in_progress':
                if task.get_last_iteration() > 0: # Try to get the first iteration metric
                    task.mark_stopped(force=True)
                    task.set_archived(True)
                    return True
            time.sleep(5)
        raise ValueError('Triggered Timeout!')
    return f'Can not find task {task}.\n\n'

if __name__ == '__main__':
    check_task_status(sys.argv[1])