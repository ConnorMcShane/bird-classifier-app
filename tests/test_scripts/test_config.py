"""Default configuration"""


class Config:


    ### Data ###
    root_dir = "/mnt/c/Users/ConnorMcShane/Documents/personal/projects/bird_classifier/"
    example_image_file = "tests/test_data/test_image.jpg"
    example_output_file = "tests/test_data/example_output.npy"
    classes_url = "https://www.gstatic.com/aihub/tfhub/labelmaps/aiy_birds_V1_labelmap.csv"

    batch_size = 6

    img_size = (224, 224)
    img_to_rgb = True
    img_devide_by_255 = True


    ### Model ###
    model_url = "https://tfhub.dev/google/aiy/vision/classifier/birds_V1/1"
    model_version = "latest"


    ### Logging ###
    log_file = "tests/test_outputs/bird_classifier_test.log"
    log_level_file = "DEBUG"
    log_level_console = "WARNING"

    wandb = False
    wandb_project = "bird_classifier"
    wandb_run_name = "testing"
    wandb_entity = "connor-mcshane"
    wandb_tags = []


    ### Metrics ###
    top_n = 3
