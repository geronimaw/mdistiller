from .trainer import BaseTrainer, CRDTrainer, DOT, CRDDOT, HROKD
trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "dot": DOT,
    "crd_dot": CRDDOT,
    "hrokd": HROKD,
}
