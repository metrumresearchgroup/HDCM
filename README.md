# HDCM

## Installing the packages

- Recommended IDE is Visual Studio Code.
- Open a julia REPL by going to `View` -> `Command Palette` then typing `Julia: Start REPL`.
- Make sure you are in the root directory (where the `Project.toml` and `Manifest.toml` files are located) then activate the julia project environment by typing `]` -> `activate .` -> `instantiate`. This will install all packages listed in the `Project.toml` file and make them available for loading whenever this project environment is activated.

## Running HDCM

The main script is `hdcm_numeric.jl` which loads the packages, reads the data from the `data` directory, runs the HDCM analysis, then saves all outputs in the `results` directory.