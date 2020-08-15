module upgradedep //@codelens("module upgradedep", "Sync vendor directory", "vendor"),codelens("module upgradedep", "Upgrade all dependencies", "upgrade_dependency")

// TODO(microsoft/vscode-go#12): Another issue. //@link(`microsoft/vscode-go#12`, `https://github.com/microsoft/vscode-go/issues/12`)

go 1.12

// TODO(golang/go#1234): Link the relevant issue. //@link(`golang/go#1234`, `https://github.com/golang/go/issues/1234`)

require example.com/extramodule v1.0.0 //@link(`example.com/extramodule`, `https://pkg.go.dev/mod/example.com/extramodule@v1.0.0`),codelens("require example.com/extramodule v1.0.0", "Upgrade dependency to v1.1.0", "upgrade_dependency")

// https://example.com/comment: Another issue. //@link(`https://example.com/comment`,`https://example.com/comment`)
