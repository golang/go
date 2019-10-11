package checker_test

import (
	"fmt"
	"go/ast"
	"io/ioutil"
	"path/filepath"
	"testing"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/analysistest"
	"golang.org/x/tools/go/analysis/internal/checker"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/testenv"
)

var from, to string

func TestApplyFixes(t *testing.T) {
	testenv.NeedsGoPackages(t)

	from = "bar"
	to = "baz"

	files := map[string]string{
		"rename/test.go": `package rename

func Foo() {
	bar := 12
	_ = bar
}

// the end
`}
	want := `package rename

func Foo() {
	baz := 12
	_ = baz
}

// the end
`

	testdata, cleanup, err := analysistest.WriteFiles(files)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(testdata, "src/rename/test.go")
	checker.Fix = true
	checker.Run([]string{"file=" + path}, []*analysis.Analyzer{analyzer})

	contents, err := ioutil.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	got := string(contents)
	if got != want {
		t.Errorf("contents of rewritten file\ngot: %s\nwant: %s", got, want)
	}

	defer cleanup()
}

var analyzer = &analysis.Analyzer{
	Name:     "rename",
	Requires: []*analysis.Analyzer{inspect.Analyzer},
	Run:      run,
}

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeFilter := []ast.Node{(*ast.Ident)(nil)}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		ident := n.(*ast.Ident)
		if ident.Name == from {
			msg := fmt.Sprintf("renaming %q to %q", from, to)
			pass.Report(analysis.Diagnostic{
				Pos:     ident.Pos(),
				End:     ident.End(),
				Message: msg,
				SuggestedFixes: []analysis.SuggestedFix{{
					Message: msg,
					TextEdits: []analysis.TextEdit{{
						Pos:     ident.Pos(),
						End:     ident.End(),
						NewText: []byte(to),
					}},
				}},
			})
		}
	})

	return nil, nil
}
