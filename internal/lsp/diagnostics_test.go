package lsp

import (
	"go/token"
	"reflect"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/lsp/protocol"
)

func TestDiagnostics(t *testing.T) {
	packagestest.TestAll(t, testDiagnostics)
}

func testDiagnostics(t *testing.T, exporter packagestest.Exporter) {
	files := packagestest.MustCopyFileTree("testdata/diagnostics")
	// TODO(rstambler): Stop hardcoding this if we have more files that don't parse.
	files["noparse/noparse.go"] = packagestest.Copy("testdata/diagnostics/noparse/noparse.go.in")
	modules := []packagestest.Module{
		{
			Name:  "golang.org/x/tools/internal/lsp",
			Files: files,
		},
	}
	exported := packagestest.Export(t, exporter, modules)
	defer exported.Cleanup()

	wants := make(map[string][]protocol.Diagnostic)
	for _, module := range modules {
		for fragment := range module.Files {
			if !strings.HasSuffix(fragment, ".go") {
				continue
			}
			filename := exporter.Filename(exported, module.Name, fragment)
			wants[filename] = []protocol.Diagnostic{}
		}
	}
	err := exported.Expect(map[string]interface{}{
		"diag": func(pos token.Position, msg string) {
			line := float64(pos.Line - 1)
			col := float64(pos.Column - 1)
			want := protocol.Diagnostic{
				Range: protocol.Range{
					Start: protocol.Position{
						Line:      line,
						Character: col,
					},
					End: protocol.Position{
						Line:      line,
						Character: col,
					},
				},
				Severity: protocol.SeverityError,
				Source:   "LSP: Go compiler",
				Message:  msg,
			}
			if _, ok := wants[pos.Filename]; ok {
				wants[pos.Filename] = append(wants[pos.Filename], want)
			} else {
				t.Errorf("unexpected filename: %v", pos.Filename)
			}
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	v := newView()
	v.config = exported.Config
	v.config.Mode = packages.LoadSyntax
	for filename, want := range wants {
		diagnostics, err := v.diagnostics(filenameToURI(filename))
		if err != nil {
			t.Fatal(err)
		}
		got := diagnostics[filename]
		sort.Slice(got, func(i int, j int) bool {
			return got[i].Range.Start.Line < got[j].Range.Start.Line
		})
		if equal := reflect.DeepEqual(want, got); !equal {
			t.Errorf("diagnostics failed for %s: (expected: %v), (got: %v)", filename, want, got)
		}
	}
}
