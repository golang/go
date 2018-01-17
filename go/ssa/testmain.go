// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// CreateTestMainPackage synthesizes a main package that runs all the
// tests of the supplied packages.
// It is closely coupled to $GOROOT/src/cmd/go/test.go and $GOROOT/src/testing.
//
// TODO(adonovan): this file no longer needs to live in the ssa package.
// Move it to ssautil.

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/types"
	"log"
	"os"
	"strings"
	"text/template"
)

// FindTests returns the Test, Benchmark, and Example functions
// (as defined by "go test") defined in the specified package,
// and its TestMain function, if any.
func FindTests(pkg *Package) (tests, benchmarks, examples []*Function, main *Function) {
	prog := pkg.Prog

	// The first two of these may be nil: if the program doesn't import "testing",
	// it can't contain any tests, but it may yet contain Examples.
	var testSig *types.Signature                              // func(*testing.T)
	var benchmarkSig *types.Signature                         // func(*testing.B)
	var exampleSig = types.NewSignature(nil, nil, nil, false) // func()

	// Obtain the types from the parameters of testing.MainStart.
	if testingPkg := prog.ImportedPackage("testing"); testingPkg != nil {
		mainStart := testingPkg.Func("MainStart")
		params := mainStart.Signature.Params()
		testSig = funcField(params.At(1).Type())
		benchmarkSig = funcField(params.At(2).Type())

		// Does the package define this function?
		//   func TestMain(*testing.M)
		if f := pkg.Func("TestMain"); f != nil {
			sig := f.Type().(*types.Signature)
			starM := mainStart.Signature.Results().At(0).Type() // *testing.M
			if sig.Results().Len() == 0 &&
				sig.Params().Len() == 1 &&
				types.Identical(sig.Params().At(0).Type(), starM) {
				main = f
			}
		}
	}

	// TODO(adonovan): use a stable order, e.g. lexical.
	for _, mem := range pkg.Members {
		if f, ok := mem.(*Function); ok &&
			ast.IsExported(f.Name()) &&
			strings.HasSuffix(prog.Fset.Position(f.Pos()).Filename, "_test.go") {

			switch {
			case testSig != nil && isTestSig(f, "Test", testSig):
				tests = append(tests, f)
			case benchmarkSig != nil && isTestSig(f, "Benchmark", benchmarkSig):
				benchmarks = append(benchmarks, f)
			case isTestSig(f, "Example", exampleSig):
				examples = append(examples, f)
			default:
				continue
			}
		}
	}
	return
}

// Like isTest, but checks the signature too.
func isTestSig(f *Function, prefix string, sig *types.Signature) bool {
	return isTest(f.Name(), prefix) && types.Identical(f.Signature, sig)
}

// Given the type of one of the three slice parameters of testing.Main,
// returns the function type.
func funcField(slice types.Type) *types.Signature {
	return slice.(*types.Slice).Elem().Underlying().(*types.Struct).Field(1).Type().(*types.Signature)
}

// isTest tells whether name looks like a test (or benchmark, according to prefix).
// It is a Test (say) if there is a character after Test that is not a lower-case letter.
// We don't want TesticularCancer.
// Plundered from $GOROOT/src/cmd/go/test.go
func isTest(name, prefix string) bool {
	if !strings.HasPrefix(name, prefix) {
		return false
	}
	if len(name) == len(prefix) { // "Test" is ok
		return true
	}
	return ast.IsExported(name[len(prefix):])
}

// CreateTestMainPackage creates and returns a synthetic "testmain"
// package for the specified package if it defines tests, benchmarks or
// executable examples, or nil otherwise.  The new package is named
// "main" and provides a function named "main" that runs the tests,
// similar to the one that would be created by the 'go test' tool.
//
// Subsequent calls to prog.AllPackages include the new package.
// The package pkg must belong to the program prog.
func (prog *Program) CreateTestMainPackage(pkg *Package) *Package {
	if pkg.Prog != prog {
		log.Fatal("Package does not belong to Program")
	}

	// Template data
	var data struct {
		Pkg                         *Package
		Tests, Benchmarks, Examples []*Function
		Main                        *Function
		Go18                        bool
	}
	data.Pkg = pkg

	// Enumerate tests.
	data.Tests, data.Benchmarks, data.Examples, data.Main = FindTests(pkg)
	if data.Main == nil &&
		data.Tests == nil && data.Benchmarks == nil && data.Examples == nil {
		return nil
	}

	// Synthesize source for testmain package.
	path := pkg.Pkg.Path() + "$testmain"
	tmpl := testmainTmpl
	if testingPkg := prog.ImportedPackage("testing"); testingPkg != nil {
		// In Go 1.8, testing.MainStart's first argument is an interface, not a func.
		data.Go18 = types.IsInterface(testingPkg.Func("MainStart").Signature.Params().At(0).Type())
	} else {
		// The program does not import "testing", but FindTests
		// returned non-nil, which must mean there were Examples
		// but no Test, Benchmark, or TestMain functions.

		// We'll simply call them from testmain.main; this will
		// ensure they don't panic, but will not check any
		// "Output:" comments.
		// (We should not execute an Example that has no
		// "Output:" comment, but it's impossible to tell here.)
		tmpl = examplesOnlyTmpl
	}
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		log.Fatalf("internal error expanding template for %s: %v", path, err)
	}
	if false { // debugging
		fmt.Fprintln(os.Stderr, buf.String())
	}

	// Parse and type-check the testmain package.
	f, err := parser.ParseFile(prog.Fset, path+".go", &buf, parser.Mode(0))
	if err != nil {
		log.Fatalf("internal error parsing %s: %v", path, err)
	}
	conf := types.Config{
		DisableUnusedImportCheck: true,
		Importer:                 importer{pkg},
	}
	files := []*ast.File{f}
	info := &types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Implicits:  make(map[ast.Node]types.Object),
		Scopes:     make(map[ast.Node]*types.Scope),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
	}
	testmainPkg, err := conf.Check(path, prog.Fset, files, info)
	if err != nil {
		log.Fatalf("internal error type-checking %s: %v", path, err)
	}

	// Create and build SSA code.
	testmain := prog.CreatePackage(testmainPkg, files, info, false)
	testmain.SetDebugMode(false)
	testmain.Build()
	testmain.Func("main").Synthetic = "test main function"
	testmain.Func("init").Synthetic = "package initializer"
	return testmain
}

// An implementation of types.Importer for an already loaded SSA program.
type importer struct {
	pkg *Package // package under test; may be non-importable
}

func (imp importer) Import(path string) (*types.Package, error) {
	if p := imp.pkg.Prog.ImportedPackage(path); p != nil {
		return p.Pkg, nil
	}
	if path == imp.pkg.Pkg.Path() {
		return imp.pkg.Pkg, nil
	}
	return nil, fmt.Errorf("not found") // can't happen
}

var testmainTmpl = template.Must(template.New("testmain").Parse(`
package main

import "io"
import "os"
import "testing"
import p {{printf "%q" .Pkg.Pkg.Path}}

{{if .Go18}}
type deps struct{}

func (deps) ImportPath() string { return "" }
func (deps) MatchString(pat, str string) (bool, error) { return true, nil }
func (deps) StartCPUProfile(io.Writer) error { return nil }
func (deps) StartTestLog(io.Writer) {}
func (deps) StopCPUProfile() {}
func (deps) StopTestLog() error { return nil }
func (deps) WriteHeapProfile(io.Writer) error { return nil }
func (deps) WriteProfileTo(string, io.Writer, int) error { return nil }

var match deps
{{else}}
func match(_, _ string) (bool, error) { return true, nil }
{{end}}

func main() {
	tests := []testing.InternalTest{
{{range .Tests}}
		{ {{printf "%q" .Name}}, p.{{.Name}} },
{{end}}
	}
	benchmarks := []testing.InternalBenchmark{
{{range .Benchmarks}}
		{ {{printf "%q" .Name}}, p.{{.Name}} },
{{end}}
	}
	examples := []testing.InternalExample{
{{range .Examples}}
		{Name: {{printf "%q" .Name}}, F: p.{{.Name}}},
{{end}}
	}
	m := testing.MainStart(match, tests, benchmarks, examples)
{{with .Main}}
	p.{{.Name}}(m)
{{else}}
	os.Exit(m.Run())
{{end}}
}

`))

var examplesOnlyTmpl = template.Must(template.New("examples").Parse(`
package main

import p {{printf "%q" .Pkg.Pkg.Path}}

func main() {
{{range .Examples}}
	p.{{.Name}}()
{{end}}
}
`))
