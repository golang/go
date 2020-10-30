package regtest

import (
	"testing"

	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/testenv"
)

const basicProxy = `
-- golang.org/x/hello@v1.2.3/go.mod --
module golang.org/x/hello

go 1.14
-- golang.org/x/hello@v1.2.3/hi/hi.go --
package hi

var Goodbye error
`

func TestInconsistentVendoring(t *testing.T) {
	testenv.NeedsGo1Point(t, 14)

	const pkgThatUsesVendoring = `
-- go.mod --
module mod.com

go 1.14

require golang.org/x/hello v1.2.3
-- go.sum --
golang.org/x/hello v1.2.3 h1:EcMp5gSkIhaTkPXp8/3+VH+IFqTpk3ZbpOhqk0Ncmho=
golang.org/x/hello v1.2.3/go.mod h1:WW7ER2MRNXWA6c8/4bDIek4Hc/+DofTrMaQQitGXcco=
-- vendor/modules.txt --
-- a/a1.go --
package a

import "golang.org/x/hello/hi"

func _() {
	_ = hi.Goodbye
	var q int // hardcode a diagnostic
}
`
	// TODO(rstambler): Remove this when golang/go#41819 is resolved.
	withOptions(
		WithModes(WithoutExperiments),
		WithProxyFiles(basicProxy),
	).run(t, pkgThatUsesVendoring, func(t *testing.T, env *Env) {
		env.OpenFile("a/a1.go")
		env.Await(
			// The editor should pop up a message suggesting that the user
			// run `go mod vendor`, along with a button to do so.
			// By default, the fake editor always accepts such suggestions,
			// so once we see the request, we can assume that `go mod vendor`
			// will be executed.
			OnceMet(
				CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidOpen), 1),
				env.DiagnosticAtRegexp("go.mod", "module mod.com"),
			),
		)
		// Apply the quickfix associated with the diagnostic.
		d := &protocol.PublishDiagnosticsParams{}
		env.Await(ReadDiagnostics("go.mod", d))
		env.ApplyQuickFixes("go.mod", d.Diagnostics)

		// Check for file changes when the command completes.
		env.Await(CompletedWork(source.CommandVendor.Title, 1))
		env.CheckForFileChanges()

		// Confirm that there is no longer any inconsistent vendoring.
		env.Await(
			DiagnosticAt("a/a1.go", 6, 5),
		)
	})
}
