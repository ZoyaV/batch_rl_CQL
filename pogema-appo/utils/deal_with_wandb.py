def fix_wandb_git():
    import os
    if "WANDB_API_KEY" in os.environ:
        os.system("rm -rf .git")
