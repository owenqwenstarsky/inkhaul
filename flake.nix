# The flake interface to llama.cpp's Nix expressions. The flake is used as a
# more discoverable entry-point, as well as a way to pin the dependencies and
# expose default outputs, including the outputs built by the CI.

# For more serious applications involving some kind of customization  you may
# want to consider consuming the overlay, or instantiating `llamaPackages`
# directly:
#
# ```nix
# pkgs.callPackage ${llama-cpp-root}/.devops/nix/scope.nix { }`
# ```

# Cf. https://jade.fyi/blog/flakes-arent-real/ for a more detailed exposition
# of the relation between Nix and the Nix Flakes.
{
  description = "llama.cpp fork for Inkling 2 Mini on Apple Silicon (Metal) and NVIDIA GPUs (CUDA)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
  };

  # For inspection, use `nix flake show` or the nix repl:
  #
  # ```bash
  # $ nix repl
  # nix-repl> :lf .
  # nix-repl> outputs.apps.aarch64-darwin.llama-quantize
  # { program = "/nix/store/00000000000000000000000000000000-llama.cpp/bin/llama-quantize"; type = "app"; }
  # ```
  outputs =
    { self, flake-parts, ... }@inputs:
    let
      # We could include the git revisions in the package names but those would
      # needlessly trigger rebuilds:
      # llamaVersion = self.dirtyShortRev or self.shortRev;

      # Nix already uses cryptographic hashes for versioning, so we'll just fix
      # the fake semver for now:
      llamaVersion = "0.0.0";
    in
    flake-parts.lib.mkFlake { inherit inputs; }

      {

        imports = [
          .devops/nix/nixpkgs-instances.nix
          .devops/nix/apps.nix
          .devops/nix/devshells.nix
        ];

        # An overlay can be used to have a more granular control over llama-cpp's
        # dependencies and configuration, than that offered by the `.override`
        # mechanism. Cf. https://nixos.org/manual/nixpkgs/stable/#chap-overlays.
        #
        # E.g. in a flake:
        # ```
        # { nixpkgs, llama-cpp, ... }:
        # let pkgs = import nixpkgs {
        #     overlays = [ (llama-cpp.overlays.default) ];
        #     system = "aarch64-darwin";
        # }; in {
        #     packages.aarch64-darwin.llama = pkgs.llamaPackages.llama-cpp;
        # }
        # ```
        #
        # Cf. https://nixos.org/manual/nix/unstable/command-ref/new-cli/nix3-flake.html?highlight=flake#flake-format
        flake.overlays.default = (
          final: prev: {
            llamaPackages = final.callPackage .devops/nix/scope.nix { inherit llamaVersion; };
            inherit (final.llamaPackages) llama-cpp;
          }
        );

        # this fork targets Apple Silicon Macs and x86_64 Linux with CUDA
        systems = [
          "aarch64-darwin"
          "x86_64-linux"
        ];

        perSystem =
          {
            config,
            lib,
            pkgs,
            pkgsCuda,
            ...
          }:
          {
            # For standardised reproducible formatting with `nix fmt`
            formatter = pkgs.nixfmt;

            # Unlike `.#packages`, legacyPackages may contain values of
            # arbitrary types (including nested attrsets) and may even throw
            # exceptions. This attribute isn't recursed into by `nix flake
            # show` either.
            #
            # You can add arbitrary scripts to `.devops/nix/scope.nix` and
            # access them as `nix build .#llamaPackages.${scriptName}` using
            # the same path you would with an overlay.
            legacyPackages = {
              llamaPackages = pkgs.callPackage .devops/nix/scope.nix { inherit llamaVersion; };
              llamaPackagesCuda = pkgsCuda.callPackage .devops/nix/scope.nix { inherit llamaVersion; };
            };

            # We don't use the overlay here so as to avoid making too many instances of nixpkgs,
            # cf. https://zimbatm.com/notes/1000-instances-of-nixpkgs
            packages =
              {
                default = config.legacyPackages.llamaPackages.llama-cpp;
                python-scripts = config.legacyPackages.llamaPackages.python-scripts;
              }
              // lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
                cuda = config.legacyPackages.llamaPackagesCuda.llama-cpp;
              };

            # Packages exposed in `.#checks` will be built by the CI and by
            # `nix flake check`.
            checks = {
              inherit (config.packages) default;
            };
          };
      };
}
