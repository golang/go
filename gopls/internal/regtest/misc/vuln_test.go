// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package misc

import (
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/tools/internal/lsp/command"
	"golang.org/x/tools/internal/lsp/protocol"
	. "golang.org/x/tools/internal/lsp/regtest"
	"golang.org/x/tools/internal/testenv"
)

func TestRunVulncheckExpError(t *testing.T) {
	const files = `
-- go.mod --
module mod.com

go 1.12
-- foo.go --
package foo
`
	Run(t, files, func(t *testing.T, env *Env) {
		cmd, err := command.NewRunVulncheckExpCommand("Run Vulncheck Exp", command.VulncheckArgs{
			Dir: "/invalid/file/url", // invalid arg
		})
		if err != nil {
			t.Fatal(err)
		}

		params := &protocol.ExecuteCommandParams{
			Command:   command.RunVulncheckExp.ID(),
			Arguments: cmd.Arguments,
		}

		response, err := env.Editor.ExecuteCommand(env.Ctx, params)
		// We want an error!
		if err == nil {
			t.Errorf("got success, want invalid file URL error: %v", response)
		}
	})
}

func TestRunVulncheckExp(t *testing.T) {
	testenv.NeedsGo1Point(t, 18)
	const files = `
-- go.mod --
module mod.com

go 1.18
-- main.go --
package main

import (
        "archive/zip"
        "fmt"
)

func main() {
        _, err := zip.OpenReader("file.zip")  // vulnerable.
        fmt.Println(err)
}
`

	cwd, _ := os.Getwd()
	WithOptions(
		EnvVars{
			// Let the analyzer read vulnerabilities data from the testdata/vulndb.
			"GOVULNDB": "file://" + filepath.Join(cwd, "testdata", "vulndb"),
			// When fetchinging stdlib package vulnerability info,
			// behave as if our go version is go1.18 for this testing.
			// The default behavior is to run `go env GOVERSION` (which isn't mutable env var).
			// See gopls/internal/vulncheck.goVersion
			// which follows the convention used in golang.org/x/vuln/cmd/govulncheck.
			"GOVERSION":                       "go1.18",
			"_GOPLS_TEST_BINARY_RUN_AS_GOPLS": "true",
		},
	).Run(t, files, func(t *testing.T, env *Env) {
		cmd, err := command.NewRunVulncheckExpCommand("Run Vulncheck Exp", command.VulncheckArgs{
			Dir: env.Sandbox.Workdir.RootURI(),
		})
		if err != nil {
			t.Fatal(err)
		}

		env.ExecuteCommand(&protocol.ExecuteCommandParams{
			Command:   command.RunVulncheckExp.ID(),
			Arguments: cmd.Arguments,
		}, nil)
		env.Await(
			CompletedWork("Checking vulnerability", 1, true),
			// TODO(hyangah): once the diagnostics are published, wait for diagnostics.
		)
	})
}
