//+build go1.15

package regtest

import (
	"testing"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/testenv"
)

func TestRegenerateCgo(t *testing.T) {
	t.Skip("This test fails in some environments: see golang.org/issues/39135")
	testenv.NeedsTool(t, "cgo")

	const workspace = `
-- go.mod --
module example.com
-- cgo.go --
package x

/*
int fortythree() { return 42; }
*/
import "C"

func Foo() {
	print(C.fortytwo())
}
`
	runner.Run(t, workspace, func(t *testing.T, env *Env) {
		// Open the file. We should have a nonexistant symbol.
		env.OpenFile("cgo.go")
		env.Await(env.DiagnosticAtRegexp("cgo.go", `C\.(fortytwo)`)) // could not determine kind of name for C.fortytwo

		// Fix the C function name. We haven't regenerated cgo, so nothing should be fixed.
		env.RegexpReplace("cgo.go", `int fortythree`, "int fortytwo")
		env.SaveBuffer("cgo.go")
		env.Await(env.DiagnosticAtRegexp("cgo.go", `C\.(fortytwo)`))

		// Regenerate cgo, fixing the diagnostic.
		lenses := env.CodeLens("cgo.go")
		var lens protocol.CodeLens
		for _, l := range lenses {
			if l.Command.Command == source.CommandRegenerateCgo {
				lens = l
			}
		}
		if _, err := env.Editor.Server.ExecuteCommand(env.Ctx, &protocol.ExecuteCommandParams{
			Command:   lens.Command.Command,
			Arguments: lens.Command.Arguments,
		}); err != nil {
			t.Fatal(err)
		}
		env.Await(EmptyDiagnostics("cgo.go"))
	})
}
