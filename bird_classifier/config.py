"""Default configuration"""


class Config:


    ### Data ###
    root_dir = "/mnt/c/Users/ConnorMcShane/Documents/personal/projects/bird_classifier/"
    classes_url = "https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv"

    batch_size = 6

    img_size = (224, 224)
    img_to_rgb = True
    img_devide_by_255 = True


    ### Model ###
    model_registry = "bird_classifier/model_registry.json"
    model_name = "aiy/birds"
    model_version = "latest"


    ### Logging ###
    log_file = "logs/bird_classifier.log"
    log_level_file = "DEBUG"
    log_level_console = "DEBUG"

    wandb = False
    wandb_project = "bird_classifier"
    wandb_run_name = "testing"
    wandb_entity = "connor-mcshane"
    wandb_tags = []


    ### Metrics ###
    top_n = 3
