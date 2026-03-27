import optuna

storage = "sqlite:///cellfoundry_cell_speed.db"
study_name = "cellfoundry_cell_speed_tgfb_chemokinesis"

# Delete only that study
optuna.delete_study(study_name=study_name, storage=storage)

