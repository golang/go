package regtest

import (
	"testing"

	"golang.org/x/tools/internal/lsp"
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
-- vendor/modules.txt --
-- a/a1.go --
package a

import "golang.org/x/hello/hi"

func _() {
	_ = hi.Goodbye
	var q int // hardcode a diagnostic
}
`
	runner.Run(t, pkgThatUsesVendoring, func(t *testing.T, env *Env) {
		env.OpenFile("a/a1.go")
		env.Await(
			// The editor should pop up a message suggesting that the user
			// run `go mod vendor`, along with a button to do so.
			// By default, the fake editor always accepts such suggestions,
			// so once we see the request, we can assume that `go mod vendor`
			// will be executed.
			OnceMet(
				CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidOpen), 1),
				ShowMessageRequest("go mod vendor"),
			),
		)
		env.CheckForFileChanges()
		env.Await(
			OnceMet(
				CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromDidChangeWatchedFiles), 1),
				DiagnosticAt("a/a1.go", 6, 5),
			),
		)
	}, WithProxyFiles(basicProxy))
}
