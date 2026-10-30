// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gotestchecker_test

import (
	"encoding/json"
	"go/analysis"
	"go/ast"
	"go/types"
	"internal/testenv"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"internal/analysis/gotestchecker"
	"internal/analysis/gotestchecker/protocol"
)

// toolEnv is the environment variable that causes the test binary to
// act as an analysis tool containing the variables of toolVars.
const toolEnv = "GOTESTCHECKER_TEST_TOOL"

const toolSymbol = "example.com/special.Analyzer"

var toolVars = []gotestchecker.Var{{Symbol: toolSymbol, Value: specialAnalyzer}}

func TestMain(m *testing.M) {
	if os.Getenv(toolEnv) != "" {
		gotestchecker.Main(toolVars)
		os.Exit(0)
	}
	os.Exit(m.Run())
}

// TestRun executes the driver as the go command would, without the
// go command: it runs the test binary as the analysis tool, first to
// query the metadata, then to analyze two packages, lib and a (which
// imports lib), using the results of the first (its export data and
// facts) as inputs to the second.
//
// TODO(adonovan): turn this into a go script integration test when we can.
func TestRun(t *testing.T) {
	testenv.MustHaveExec(t)
	dir := t.TempDir()
	file := func(name string) string { return filepath.Join(dir, name) }
	writeFile := func(name, content string) string {
		t.Helper()
		if err := os.WriteFile(file(name), []byte(content), 0666); err != nil {
			t.Fatal(err)
		}
		return file(name)
	}

	// tool runs the analysis tool with the specified arguments.
	tool := func(args ...string) {
		t.Helper()
		cmd := testenv.Command(t, testenv.Executable(t), args...)
		cmd.Env = append(os.Environ(), toolEnv+"=1")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("%v failed: %v\n%s", cmd, err, out)
		}
	}

	// Query the metadata for the analyzer's index.
	tool("-describe=" + file("metadata.json"))
	data, err := os.ReadFile(file("metadata.json"))
	if err != nil {
		t.Fatal(err)
	}
	var md protocol.Metadata
	if err := json.Unmarshal(data, &md); err != nil {
		t.Fatal(err)
	}
	info := md.Var(toolSymbol)
	if info == nil || info.Error != "" || len(info.Members) != 1 {
		t.Fatalf("metadata for %s: %+v", toolSymbol, info)
	}
	special := info.Members[0].Analyzer
	if a := md.Analyzers[special]; a.Name != "special" || !a.Facts {
		t.Fatalf("analyzer %d: got %+v, want special, which uses facts", special, a)
	}

	// analyze runs the tool on the package described by req,
	// and returns the pass of the special analyzer.
	analyze := func(req *protocol.AnalyzeRequest) *protocol.Pass {
		t.Helper()
		data, err := json.Marshal(req)
		if err != nil {
			t.Fatal(err)
		}
		tool(writeFile(req.ID+".json", string(data)))
		pass, err := protocol.ReadPass(req.Outputs[special])
		if err != nil {
			t.Fatal(err)
		}
		if pass.Failure != nil || pass.Skipped {
			t.Fatalf("%s: pass failed (%v) or was skipped (%t)", req.ID, pass.Failure, pass.Skipped)
		}
		return pass
	}

	// Analyze lib, whose diagnostics are not reported,
	// for its export data and facts.
	libPass := analyze(&protocol.AnalyzeRequest{
		ID:          "lib",
		ImportPath:  "lib",
		Dir:         dir,
		GoFiles:     {writeFile("lib.go", "package lib\n\nfunc SpecialF() {}\n\nfunc G() { SpecialF() }\n")},
		GoVersion:   "go1.28",
		GOARCH:      runtime.GOARCH,
		TypeImports: {},
		TypeExport:  file("lib.export"),
		Facts:       {},
		Outputs:     {special: file("lib.pass")},
	})
	if len(libPass.Facts) == 0 {
		t.Errorf("lib: no facts")
	}
	if len(libPass.Diagnostics) > 0 {
		t.Errorf("lib: unexpected diagnostics: %v", libPass.Diagnostics)
	}

	// Analyze a, using the results of lib.
	aPass := analyze(&protocol.AnalyzeRequest{
		ID:          "a",
		ImportPath:  "a",
		Dir:         dir,
		GoFiles:     {writeFile("a.go", "package a\n\nimport \"lib\"\n\nfunc f() {\n\tlib.SpecialF()\n\tlib.G()\n}\n")},
		ImportMap:   {"lib": "lib"},
		GoVersion:   "go1.28",
		GOARCH:      runtime.GOARCH,
		TypeImports: {"lib": file("lib.export")},
		Facts:       {special: {"lib": file("lib.pass")}},
		Outputs:     {special: file("a.pass")},
		Report:      {special},
	})
	if len(aPass.Diagnostics) != 1 {
		t.Fatalf("a: got diagnostics %v, want one", aPass.Diagnostics)
	}
	d := aPass.Diagnostics[0]
	if want := "call of special function lib.SpecialF"; d.Message != want {
		t.Errorf("a: got diagnostic %q, want %q", d.Message, want)
	}
	if want := "a.go:6:2"; !strings.HasSuffix(d.Posn, want) {
		t.Errorf("a: got diagnostic at %s, want %s", d.Posn, want)
	}
}

// specialAnalyzer reports calls of functions whose names start with
// "Special", using facts to identify such functions in other packages.
var specialAnalyzer = &analysis.Analyzer{
	Name:      "special",
	Doc:       "report calls of special functions",
	FactTypes: []analysis.Fact{new(isSpecial)},
	Run: func(pass *analysis.Pass) (any, error) {
		for _, f := range pass.Files {
			for _, decl := range f.Decls {
				if decl, ok := decl.(*ast.FuncDecl); ok && strings.HasPrefix(decl.Name.Name, "Special") {
					pass.ExportObjectFact(pass.TypesInfo.Defs[decl.Name], &isSpecial{true})
				}
			}
		}
		for _, f := range pass.Files {
			ast.Inspect(f, func(n ast.Node) bool {
				if sel, ok := n.(*ast.SelectorExpr); ok {
					if fn, ok := pass.TypesInfo.Uses[sel.Sel].(*types.Func); ok && pass.ImportObjectFact(fn, new(isSpecial)) {
						pass.ReportRangef(sel, "call of special function %s.%s", fn.Pkg().Name(), fn.Name())
					}
				}
				return true
			})
		}
		return nil, nil
	},
}

// An isSpecial fact marks a special function.
type isSpecial struct{ Special bool } // (gob requires an exported field)

func (*isSpecial) AFact() {}
