# Install pre-built version of llama.cpp

> [!NOTE]
> These packages are builds of upstream llama.cpp, not of this fork. They do not include LongHaul or the Inkling 2 changes. To run Inkling 2 Mini with LongHaul, build this repository from source (see [build.md](build.md)).

| Install via | Mac  |
|-------------|------|
| conda-forge | yes  |
| Homebrew    | yes  |
| MacPorts    | yes  |
| Nix         | yes  |

## conda-forge (Mac)

conda-forge provides builds for Apple Metal (macOS).

```sh
conda install -c conda-forge llama.cpp
```

```sh
mamba install -c conda-forge llama.cpp
```

```sh
# Project-local installation
pixi add llama.cpp

# Global installation
pixi global install llama.cpp
```

This distribution is managed on [`conda-forge/llama.cpp-feedstock`](https://github.com/conda-forge/llama.cpp-feedstock/).

Shall you have any problems, please open an issue on [its issue tracker](https://github.com/conda-forge/llama.cpp-feedstock/issues).

## Homebrew (Mac)

```sh
brew install llama.cpp
```

The formula is automatically updated with new `llama.cpp` releases. More info: https://github.com/ggml-org/llama.cpp/discussions/7668

## MacPorts (Mac)

```sh
sudo port install llama.cpp
```

See also: https://ports.macports.org/port/llama.cpp/details/

## Nix (Mac)

```sh
nix profile install nixpkgs#llama-cpp
```

For flake enabled installs.

Or

```sh
nix-env --file '<nixpkgs>' --install --attr llama-cpp
```

For non-flake enabled installs.

This expression is automatically updated within the [nixpkgs repo](https://github.com/NixOS/nixpkgs/blob/nixos-24.05/pkgs/by-name/ll/llama-cpp/package.nix#L164).
