{
  description = "Development shell with Rust and Go";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {inherit system;};
        deps = with pkgs; [
          (python3.withPackages (python-pkgs: [python-pkgs.requests]))
          go
          rustup
          cargo
          rustc
        ];
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = deps;
          shellHook = ''
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${builtins.toString (pkgs.lib.makeLibraryPath deps)}";
          '';
        };
      }
    );
}
