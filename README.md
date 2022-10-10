# hackathon QuantumBlack, McKinsey + Binet IA, École Polytechnique

The problem of the hackathon consisted in classifying satellite images to check whether they had silos in them, to
aid a certain start-up know where to place new silos. As a bonus task, the goal was to segment images into the classes
"silo" or "not silo".

To create the environment, run ``conda env create -f environment.yml``.

To track training, ``tensorboard --logdir lightning_logs --bind_all``.

The dataset given is in ``ai_ready/``. The test dataset is given in ``test_to_send/``.

All models built are in ``models/``
To choose which model to train, change line ``from models.model_unet import HackathonModel`` in ``run_training.py``.
To launch training, ``python run_training.py -v version_name``.

The checkpoints for our best models are in ``model_weights/``.
The best model found was ``model_efficient`` for classification. The only model made for segmentation was 
``model_unet``.

To run evaluation on a .csv with the same format as the ones found inside the datasets, run ``python run_eval -p 
path_to_csv``.

