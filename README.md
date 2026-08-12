# Structure

> **Data**
> contains processed population_synthesis runs

> **Analysis**
> contains partial scripts and notebooks. Unfinished work

> **figure**
> contains finalized figures

> **population_synthesis**
> contains the popsynth runs and post-processing.




## Population synthesis scripts

- `optical_depth.py` does the optical depth calculation based on Langer 1989. 
https://ui.adsabs.harvard.edu/abs/1989A%26A...210...93L/abstract
- `extract_WR_phases.py` extracts the WR phases based on a luminosity limit.
- `extracT_WR_optical_depth.py` calculated the optical depth and adds it to the "WR phase"


run `python extract_WR_optical_depth.py` to get the optical depth limits