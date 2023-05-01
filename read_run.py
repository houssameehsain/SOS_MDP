import wandb
api = wandb.Api(timeout=200)

# run is specified by <entity>/<project>/<run_id>
run = api.run("hehsain/sos-mdp/nlf9c6os")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
metrics_dataframe.to_csv("metrics.csv")
