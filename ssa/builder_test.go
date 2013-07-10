package ssa_test

import (
	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/importer"
	"code.google.com/p/go.tools/ssa"
	"go/ast"
	"go/parser"
	"strings"
	"testing"
)

func isEmpty(f *ssa.Function) bool { return f.Blocks == nil }

// Tests that programs partially loaded from gc object files contain
// functions with no code for the external portions, but are otherwise ok.
func TestExternalPackages(t *testing.T) {
	test := `
package main

import (
	"bytes"
	"io"
	"testing"
)

func main() {
        var t testing.T
	t.Parallel()    // static call to external declared method
        t.Fail()        // static call to synthetic promotion wrapper
        testing.Short() // static call to external package-level function

        var w io.Writer = new(bytes.Buffer)
        w.Write(nil)    // interface invoke of external declared method
}
`
	imp := importer.New(new(importer.Context)) // no Loader; uses GC importer

	f, err := parser.ParseFile(imp.Fset, "<input>", test, parser.DeclarationErrors)
	if err != nil {
		t.Errorf("parse error: %s", err)
		return
	}

	info, err := imp.CreateSourcePackage("main", []*ast.File{f})
	if err != nil {
		t.Error(err.Error())
		return
	}

	prog := ssa.NewProgram(imp.Fset, ssa.SanityCheckFunctions)
	prog.CreatePackages(imp)
	mainPkg := prog.Package(info.Pkg)
	mainPkg.Build()

	// Only the main package and its immediate dependencies are loaded.
	deps := []string{"bytes", "io", "testing"}
	if len(prog.PackagesByPath) != 1+len(deps) {
		t.Errorf("unexpected set of loaded packages: %q", prog.PackagesByPath)
	}
	for _, path := range deps {
		pkg, _ := prog.PackagesByPath[path]
		if pkg == nil {
			t.Errorf("package not loaded: %q", path)
			continue
		}

		// External packages should have no function bodies (except for wrappers).
		isExt := pkg != mainPkg

		// init()
		if isExt && !isEmpty(pkg.init) {
			t.Errorf("external package %s has non-empty init", pkg)
		} else if !isExt && isEmpty(pkg.init) {
			t.Errorf("main package %s has empty init", pkg)
		}

		for _, mem := range pkg.Members {
			switch mem := mem.(type) {
			case *ssa.Function:
				// Functions at package level.
				if isExt && !isEmpty(mem) {
					t.Errorf("external function %s is non-empty", mem)
				} else if !isExt && isEmpty(mem) {
					t.Errorf("function %s is empty", mem)
				}

			case *ssa.Type:
				// Methods of named types T.
				// (In this test, all exported methods belong to *T not T.)
				if !isExt {
					t.Fatalf("unexpected name type in main package: %s", mem)
				}
				for _, m := range prog.MethodSet(types.NewPointer(mem.Type())) {
					// For external types, only synthetic wrappers have code.
					expExt := !strings.Contains(m.Synthetic, "wrapper")
					if expExt && !isEmpty(m) {
						t.Errorf("external method %s is non-empty: %s",
							m, m.Synthetic)
					} else if !expExt && isEmpty(m) {
						t.Errorf("method function %s is empty: %s",
							m, m.Synthetic)
					}
				}
			}
		}
	}

	expectedCallee := []string{
		"(*testing.T).Parallel",
		"(*testing.T).Fail",
		"testing.Short",
		"N/A",
	}
	callNum := 0
	for _, b := range mainPkg.Func("main").Blocks {
		for _, instr := range b.Instrs {
			switch instr := instr.(type) {
			case ssa.CallInstruction:
				call := instr.Common()
				if want := expectedCallee[callNum]; want != "N/A" {
					got := call.StaticCallee().String()
					if want != got {
						t.Errorf("%dth call from main.main: got callee %s, want %s",
							callNum, got, want)
					}
				}
				callNum++
			}
		}
	}
	if callNum != 4 {
		t.Errorf("in main.main: got %d calls, want %d", callNum, 4)
	}
}
