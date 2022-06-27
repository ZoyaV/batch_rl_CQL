from gym import register

register(
    id="POMAPF-v0",
    entry_point="pomapf.env:make_pomapf",
)
