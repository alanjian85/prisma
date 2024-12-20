{
  description = "Nix development environment of Prisma";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
  };

  outputs = { nixpkgs, ... }@inputs: let
    system = "x86_64-linux";
    pkgs =  import nixpkgs {
      inherit system;
    };
  in {
    devShells."${system}".default = pkgs.mkShell rec {
      packages = with pkgs; [
        cargo
	clippy
        fbida
	rustfmt
        vulkan-loader
      ];
      shellHook = ''
        export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:${builtins.toString (pkgs.lib.makeLibraryPath packages)}";
      '';
    };
  };
}
