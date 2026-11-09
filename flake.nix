{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # https://github.com/NixOS/nixpkgs/issues/519484
        patchedQuarto = pkgs.quarto.overrideAttrs (oldAttrs: {
          postPatch = (oldAttrs.postPatch or "") + ''
            substituteInPlace bin/quarto.js \
              --replace-fail "syntax-highlighting" "highlight-style"
          '';
        });

        pythonEnv = pkgs.python3.withPackages (
          ps: with ps; [
            jupyter
            ipykernel
            nbformat
            nbclient
            pyyaml
          ]
        );
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            patchedQuarto
            pkgs.typst
            pkgs.tinymist
            pkgs.source-sans
            pkgs.source-sans-pro
            pkgs.roboto
            pkgs.font-awesome
            pythonEnv
          ];

          TYPST_FONT_PATHS = pkgs.lib.makeSearchPathOutput "share/fonts" "" [
            pkgs.source-sans
            pkgs.source-sans-pro
            pkgs.roboto
            pkgs.font-awesome
          ];

          QUARTO_PYTHON = "${pythonEnv}/bin/python";

          shellHook = ''
            echo "Website dev env ready"
            echo "  quarto: $(quarto --version)"
            echo "  typst:  $(typst --version)"
          '';
        };
      }
    );
}
