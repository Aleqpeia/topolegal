{
  inputs = {
    nixpkgs.url = "github:cachix/devenv-nixpkgs/rolling";
    nixpkgs-python.url = "github:cachix/nixpkgs-python";
    devenv.url = "github:cachix/devenv/latest";
  };

  outputs = {
    self,
    nixpkgs,
    devenv,
    ...
  } @ inputs: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
  in {
    devShells.${system}.default = devenv.lib.mkShell {
      inherit inputs pkgs;
      modules = [
        (
          {
            pkgs,
            config,
            lib,
            ...
          }: {
            # https://devenv.sh/packages/
            packages = [
              config.languages.python.package.pkgs.pjsua2
              pkgs.zlib
              pkgs.gcc
              pkgs.ninja
              pkgs.python313Packages.spacy
            ];

            # https://devenv.sh/languages/
            languages.python = {
              enable = true;

              poetry = {
                enable = true;
                activate.enable = true;
                install = {
                  enable = true;
                  installRootPackage = true;
                  onlyInstallRootPackage = false;
                  verbosity = "more";
                };
              };
            };

            # https://devenv.sh/pre-commit-hooks/
            git-hooks.hooks = {
              alejandra.enable = true;
              black.enable = true;
            };

            # https://devenv.sh/enter-shell/
            enterShell = ''
              echo "Python version: $(python --version)"
              echo "Poetry version: $(poetry --version)"
            '';

            # Environment variables
            env = {
              GOOGLE_APPLICATION_CREDENTIALS = "/home/efyis/legalgnn.json";
              PROJECT_ID = "lab-test-project-1-305710";
            };
          }
        )
      ];
    };
  };
}
