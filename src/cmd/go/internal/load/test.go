// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/build"
	"go/doc"
	"go/parser"
	"go/token"
	"internal/lazytemplate"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"

	"cmd/go/internal/str"
	"cmd/go/internal/trace"
)

var TestMainDeps = []string{
	// Dependencies for testmain.
	"os",
	"reflect",
	"testing",
	"testing/internal/testdeps",
}

type TestCover struct {
	Mode     string
	Local    bool
	Pkgs     []*Package
	Paths    []string
	Vars     []coverInfo
	DeclVars func(*Package, ...string) map[string]*CoverVar
}

// TestPackagesFor is like TestPackagesAndErrors but it returns
// an error if the test packages or their dependencies have errors.
// Only test packages without errors are returned.
func TestPackagesFor(ctx context.Context, p *Package, cover *TestCover) (pmain, ptest, pxtest *Package, err error) {
	pmain, ptest, pxtest = TestPackagesAndErrors(ctx, p, cover)
	for _, p1 := range []*Package{ptest, pxtest, pmain} {
		if p1 == nil {
			// pxtest may be nil
			continue
		}
		if p1.Error != nil {
			err = p1.Error
			break
		}
		if len(p1.DepsErrors) > 0 {
			perr := p1.DepsErrors[0]
			err = perr
			break
		}
	}
	if pmain.Error != nil || len(pmain.DepsErrors) > 0 {
		pmain = nil
	}
	if ptest.Error != nil || len(ptest.DepsErrors) > 0 {
		ptest = nil
	}
	if pxtest != nil && (pxtest.Error != nil || len(pxtest.DepsErrors) > 0) {
		pxtest = nil
	}
	return pmain, ptest, pxtest, err
}

// TestPackagesAndErrors returns three packages:
//	- pmain, the package main corresponding to the test binary (running tests in ptest and pxtest).
//	- ptest, the package p compiled with added "package p" test files.
//	- pxtest, the result of compiling any "package p_test" (external) test files.
//
// If the package has no "package p_test" test files, pxtest will be nil.
// If the non-test compilation of package p can be reused
// (for example, if there are no "package p" test files and
// package p need not be instrumented for coverage or any other reason),
// then the returned ptest == p.
//
// An error is returned if the testmain source cannot be completely generated
// (for example, due to a syntax error in a test file). No error will be
// returned for errors loading packages, but the Error or DepsError fields
// of the returned packages may be set.
//
// The caller is expected to have checked that len(p.TestGoFiles)+len(p.XTestGoFiles) > 0,
// or else there's no point in any of this.
func TestPackagesAndErrors(ctx context.Context, p *Package, cover *TestCover) (pmain, ptest, pxtest *Package) {
	ctx, span := trace.StartSpan(ctx, "load.TestPackagesAndErrors")
	defer span.Done()

	pre := newPreload()
	defer pre.flush()
	allImports := append([]string{}, p.TestImports...)
	allImports = append(allImports, p.XTestImports...)
	pre.preloadImports(allImports, p.Internal.Build)

	var ptestErr, pxtestErr *PackageError
	var imports, ximports []*Package
	var stk ImportStack
	var testEmbed, xtestEmbed map[string][]string
	stk.Push(p.ImportPath + " (test)")
	rawTestImports := str.StringList(p.TestImports)
	for i, path := range p.TestImports {
		p1 := loadImport(ctx, pre, path, p.Dir, p, &stk, p.Internal.Build.TestImportPos[path], ResolveImport)
		if str.Contains(p1.Deps, p.ImportPath) || p1.ImportPath == p.ImportPath {
			// Same error that loadPackage returns (via reusePackage) in pkg.go.
			// Can't change that code, because that code is only for loading the
			// non-test copy of a package.
			ptestErr = &PackageError{
				ImportStack:   testImportStack(stk[0], p1, p.ImportPath),
				Err:           errors.New("import cycle not allowed in test"),
				IsImportCycle: true,
			}
		}
		p.TestImports[i] = p1.ImportPath
		imports = append(imports, p1)
	}
	var err error
	p.TestEmbedFiles, testEmbed, err = resolveEmbed(p.Dir, p.TestEmbedPatterns)
	if err != nil && ptestErr == nil {
		ptestErr = &PackageError{
			ImportStack: stk.Copy(),
			Err:         err,
		}
		embedErr := err.(*EmbedError)
		ptestErr.setPos(p.Internal.Build.TestEmbedPatternPos[embedErr.Pattern])
	}
	stk.Pop()

	stk.Push(p.ImportPath + "_test")
	pxtestNeedsPtest := false
	rawXTestImports := str.StringList(p.XTestImports)
	for i, path := range p.XTestImports {
		p1 := loadImport(ctx, pre, path, p.Dir, p, &stk, p.Internal.Build.XTestImportPos[path], ResolveImport)
		if p1.ImportPath == p.ImportPath {
			pxtestNeedsPtest = true
		} else {
			ximports = append(ximports, p1)
		}
		p.XTestImports[i] = p1.ImportPath
	}
	p.XTestEmbedFiles, xtestEmbed, err = resolveEmbed(p.Dir, p.XTestEmbedPatterns)
	if err != nil && pxtestErr == nil {
		pxtestErr = &PackageError{
			ImportStack: stk.Copy(),
			Err:         err,
		}
		embedErr := err.(*EmbedError)
		pxtestErr.setPos(p.Internal.Build.XTestEmbedPatternPos[embedErr.Pattern])
	}
	stk.Pop()

	// Test package.
	if len(p.TestGoFiles) > 0 || p.Name == "main" || cover != nil && cover.Local {
		ptest = new(Package)
		*ptest = *p
		ptest.Error = ptestErr
		ptest.ForTest = p.ImportPath
		ptest.GoFiles = nil
		ptest.GoFiles = append(ptest.GoFiles, p.GoFiles...)
		ptest.GoFiles = append(ptest.GoFiles, p.TestGoFiles...)
		ptest.Target = ""
		// Note: The preparation of the vet config requires that common
		// indexes in ptest.Imports and ptest.Internal.RawImports
		// all line up (but RawImports can be shorter than the others).
		// That is, for 0 ≤ i < len(RawImports),
		// RawImports[i] is the import string in the program text, and
		// Imports[i] is the expanded import string (vendoring applied or relative path expanded away).
		// Any implicitly added imports appear in Imports and Internal.Imports
		// but not RawImports (because they were not in the source code).
		// We insert TestImports, imports, and rawTestImports at the start of
		// these lists to preserve the alignment.
		// Note that p.Internal.Imports may not be aligned with p.Imports/p.Internal.RawImports,
		// but we insert at the beginning there too just for consistency.
		ptest.Imports = str.StringList(p.TestImports, p.Imports)
		ptest.Internal.Imports = append(imports, p.Internal.Imports...)
		ptest.Internal.RawImports = str.StringList(rawTestImports, p.Internal.RawImports)
		ptest.Internal.ForceLibrary = true
		ptest.Internal.BuildInfo = ""
		ptest.Internal.Build = new(build.Package)
		*ptest.Internal.Build = *p.Internal.Build
		m := map[string][]token.Position{}
		for k, v := range p.Internal.Build.ImportPos {
			m[k] = append(m[k], v...)
		}
		for k, v := range p.Internal.Build.TestImportPos {
			m[k] = append(m[k], v...)
		}
		ptest.Internal.Build.ImportPos = m
		if testEmbed == nil && len(p.Internal.Embed) > 0 {
			testEmbed = map[string][]string{}
		}
		for k, v := range p.Internal.Embed {
			testEmbed[k] = v
		}
		ptest.Internal.Embed = testEmbed
		ptest.EmbedFiles = str.StringList(p.EmbedFiles, p.TestEmbedFiles)
		ptest.collectDeps()
	} else {
		ptest = p
	}

	// External test package.
	if len(p.XTestGoFiles) > 0 {
		pxtest = &Package{
			PackagePublic: PackagePublic{
				Name:       p.Name + "_test",
				ImportPath: p.ImportPath + "_test",
				Root:       p.Root,
				Dir:        p.Dir,
				Goroot:     p.Goroot,
				GoFiles:    p.XTestGoFiles,
				Imports:    p.XTestImports,
				ForTest:    p.ImportPath,
				Module:     p.Module,
				Error:      pxtestErr,
				EmbedFiles: p.XTestEmbedFiles,
			},
			Internal: PackageInternal{
				LocalPrefix: p.Internal.LocalPrefix,
				Build: &build.Package{
					ImportPos: p.Internal.Build.XTestImportPos,
				},
				Imports:    ximports,
				RawImports: rawXTestImports,

				Asmflags:   p.Internal.Asmflags,
				Gcflags:    p.Internal.Gcflags,
				Ldflags:    p.Internal.Ldflags,
				Gccgoflags: p.Internal.Gccgoflags,
				Embed:      xtestEmbed,
			},
		}
		if pxtestNeedsPtest {
			pxtest.Internal.Imports = append(pxtest.Internal.Imports, ptest)
		}
		pxtest.collectDeps()
	}

	// Build main package.
	pmain = &Package{
		PackagePublic: PackagePublic{
			Name:       "main",
			Dir:        p.Dir,
			GoFiles:    []string{"_testmain.go"},
			ImportPath: p.ImportPath + ".test",
			Root:       p.Root,
			Imports:    str.StringList(TestMainDeps),
			Module:     p.Module,
		},
		Internal: PackageInternal{
			Build:      &build.Package{Name: "main"},
			BuildInfo:  p.Internal.BuildInfo,
			Asmflags:   p.Internal.Asmflags,
			Gcflags:    p.Internal.Gcflags,
			Ldflags:    p.Internal.Ldflags,
			Gccgoflags: p.Internal.Gccgoflags,
		},
	}

	// The generated main also imports testing, regexp, and os.
	// Also the linker introduces implicit dependencies reported by LinkerDeps.
	stk.Push("testmain")
	deps := TestMainDeps // cap==len, so safe for append
	for _, d := range LinkerDeps(p) {
		deps = append(deps, d)
	}
	for _, dep := range deps {
		if dep == ptest.ImportPath {
			pmain.Internal.Imports = append(pmain.Internal.Imports, ptest)
		} else {
			p1 := loadImport(ctx, pre, dep, "", nil, &stk, nil, 0)
			pmain.Internal.Imports = append(pmain.Internal.Imports, p1)
		}
	}
	stk.Pop()

	if cover != nil && cover.Pkgs != nil {
		// Add imports, but avoid duplicates.
		seen := map[*Package]bool{p: true, ptest: true}
		for _, p1 := range pmain.Internal.Imports {
			seen[p1] = true
		}
		for _, p1 := range cover.Pkgs {
			if !seen[p1] {
				seen[p1] = true
				pmain.Internal.Imports = append(pmain.Internal.Imports, p1)
			}
		}
	}

	allTestImports := make([]*Package, 0, len(pmain.Internal.Imports)+len(imports)+len(ximports))
	allTestImports = append(allTestImports, pmain.Internal.Imports...)
	allTestImports = append(allTestImports, imports...)
	allTestImports = append(allTestImports, ximports...)
	setToolFlags(allTestImports...)

	// Do initial scan for metadata needed for writing _testmain.go
	// Use that metadata to update the list of imports for package main.
	// The list of imports is used by recompileForTest and by the loop
	// afterward that gathers t.Cover information.
	t, err := loadTestFuncs(ptest)
	if err != nil && pmain.Error == nil {
		pmain.setLoadPackageDataError(err, p.ImportPath, &stk, nil)
	}
	t.Cover = cover
	if len(ptest.GoFiles)+len(ptest.CgoFiles) > 0 {
		pmain.Internal.Imports = append(pmain.Internal.Imports, ptest)
		pmain.Imports = append(pmain.Imports, ptest.ImportPath)
		t.ImportTest = true
	}
	if pxtest != nil {
		pmain.Internal.Imports = append(pmain.Internal.Imports, pxtest)
		pmain.Imports = append(pmain.Imports, pxtest.ImportPath)
		t.ImportXtest = true
	}
	pmain.collectDeps()

	// Sort and dedup pmain.Imports.
	// Only matters for go list -test output.
	sort.Strings(pmain.Imports)
	w := 0
	for _, path := range pmain.Imports {
		if w == 0 || path != pmain.Imports[w-1] {
			pmain.Imports[w] = path
			w++
		}
	}
	pmain.Imports = pmain.Imports[:w]
	pmain.Internal.RawImports = str.StringList(pmain.Imports)

	// Replace pmain's transitive dependencies with test copies, as necessary.
	recompileForTest(pmain, p, ptest, pxtest)

	// Should we apply coverage analysis locally,
	// only for this package and only for this test?
	// Yes, if -cover is on but -coverpkg has not specified
	// a list of packages for global coverage.
	if cover != nil && cover.Local {
		ptest.Internal.CoverMode = cover.Mode
		var coverFiles []string
		coverFiles = append(coverFiles, ptest.GoFiles...)
		coverFiles = append(coverFiles, ptest.CgoFiles...)
		ptest.Internal.CoverVars = cover.DeclVars(ptest, coverFiles...)
	}

	for _, cp := range pmain.Internal.Imports {
		if len(cp.Internal.CoverVars) > 0 {
			t.Cover.Vars = append(t.Cover.Vars, coverInfo{cp, cp.Internal.CoverVars})
		}
	}

	data, err := formatTestmain(t)
	if err != nil && pmain.Error == nil {
		pmain.Error = &PackageError{Err: err}
	}
	if data != nil {
		pmain.Internal.TestmainGo = &data
	}

	return pmain, ptest, pxtest
}

func testImportStack(top string, p *Package, target string) []string {
	stk := []string{top, p.ImportPath}
Search:
	for p.ImportPath != target {
		for _, p1 := range p.Internal.Imports {
			if p1.ImportPath == target || str.Contains(p1.Deps, target) {
				stk = append(stk, p1.ImportPath)
				p = p1
				continue Search
			}
		}
		// Can't happen, but in case it does...
		stk = append(stk, "<lost path to cycle>")
		break
	}
	return stk
}

// recompileForTest copies and replaces certain packages in pmain's dependency
// graph. This is necessary for two reasons. First, if ptest is different than
// preal, packages that import the package under test should get ptest instead
// of preal. This is particularly important if pxtest depends on functionality
// exposed in test sources in ptest. Second, if there is a main package
// (other than pmain) anywhere, we need to set p.Internal.ForceLibrary and
// clear p.Internal.BuildInfo in the test copy to prevent link conflicts.
// This may happen if both -coverpkg and the command line patterns include
// multiple main packages.
func recompileForTest(pmain, preal, ptest, pxtest *Package) {
	// The "test copy" of preal is ptest.
	// For each package that depends on preal, make a "test copy"
	// that depends on ptest. And so on, up the dependency tree.
	testCopy := map[*Package]*Package{preal: ptest}
	for _, p := range PackageList([]*Package{pmain}) {
		if p == preal {
			continue
		}
		// Copy on write.
		didSplit := p == pmain || p == pxtest
		split := func() {
			if didSplit {
				return
			}
			didSplit = true
			if testCopy[p] != nil {
				panic("recompileForTest loop")
			}
			p1 := new(Package)
			testCopy[p] = p1
			*p1 = *p
			p1.ForTest = preal.ImportPath
			p1.Internal.Imports = make([]*Package, len(p.Internal.Imports))
			copy(p1.Internal.Imports, p.Internal.Imports)
			p1.Imports = make([]string, len(p.Imports))
			copy(p1.Imports, p.Imports)
			p = p1
			p.Target = ""
			p.Internal.BuildInfo = ""
			p.Internal.ForceLibrary = true
		}

		// Update p.Internal.Imports to use test copies.
		for i, imp := range p.Internal.Imports {
			if p1 := testCopy[imp]; p1 != nil && p1 != imp {
				split()
				p.Internal.Imports[i] = p1
			}
		}

		// Force main packages the test imports to be built as libraries.
		// Normal imports of main packages are forbidden by the package loader,
		// but this can still happen if -coverpkg patterns include main packages:
		// covered packages are imported by pmain. Linking multiple packages
		// compiled with '-p main' causes duplicate symbol errors.
		// See golang.org/issue/30907, golang.org/issue/34114.
		if p.Name == "main" && p != pmain && p != ptest {
			split()
		}
	}
}

// isTestFunc tells whether fn has the type of a testing function. arg
// specifies the parameter type we look for: B, M or T.
func isTestFunc(fn *ast.FuncDecl, arg string) bool {
	if fn.Type.Results != nil && len(fn.Type.Results.List) > 0 ||
		fn.Type.Params.List == nil ||
		len(fn.Type.Params.List) != 1 ||
		len(fn.Type.Params.List[0].Names) > 1 {
		return false
	}
	ptr, ok := fn.Type.Params.List[0].Type.(*ast.StarExpr)
	if !ok {
		return false
	}
	// We can't easily check that the type is *testing.M
	// because we don't know how testing has been imported,
	// but at least check that it's *M or *something.M.
	// Same applies for B and T.
	if name, ok := ptr.X.(*ast.Ident); ok && name.Name == arg {
		return true
	}
	if sel, ok := ptr.X.(*ast.SelectorExpr); ok && sel.Sel.Name == arg {
		return true
	}
	return false
}

// isTest tells whether name looks like a test (or benchmark, according to prefix).
// It is a Test (say) if there is a character after Test that is not a lower-case letter.
// We don't want TesticularCancer.
func isTest(name, prefix string) bool {
	if !strings.HasPrefix(name, prefix) {
		return false
	}
	if len(name) == len(prefix) { // "Test" is ok
		return true
	}
	rune, _ := utf8.DecodeRuneInString(name[len(prefix):])
	return !unicode.IsLower(rune)
}

type coverInfo struct {
	Package *Package
	Vars    map[string]*CoverVar
}

// loadTestFuncs returns the testFuncs describing the tests that will be run.
// The returned testFuncs is always non-nil, even if an error occurred while
// processing test files.
func loadTestFuncs(ptest *Package) (*testFuncs, error) {
	t := &testFuncs{
		Package: ptest,
	}
	var err error
	for _, file := range ptest.TestGoFiles {
		if lerr := t.load(filepath.Join(ptest.Dir, file), "_test", &t.ImportTest, &t.NeedTest); lerr != nil && err == nil {
			err = lerr
		}
	}
	for _, file := range ptest.XTestGoFiles {
		if lerr := t.load(filepath.Join(ptest.Dir, file), "_xtest", &t.ImportXtest, &t.NeedXtest); lerr != nil && err == nil {
			err = lerr
		}
	}
	return t, err
}

// formatTestmain returns the content of the _testmain.go file for t.
func formatTestmain(t *testFuncs) ([]byte, error) {
	var buf bytes.Buffer
	if err := testmainTmpl.Execute(&buf, t); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

type testFuncs struct {
	Tests       []testFunc
	Benchmarks  []testFunc
	Examples    []testFunc
	TestMain    *testFunc
	Package     *Package
	ImportTest  bool
	NeedTest    bool
	ImportXtest bool
	NeedXtest   bool
	Cover       *TestCover
}

// ImportPath returns the import path of the package being tested, if it is within GOPATH.
// This is printed by the testing package when running benchmarks.
func (t *testFuncs) ImportPath() string {
	pkg := t.Package.ImportPath
	if strings.HasPrefix(pkg, "_/") {
		return ""
	}
	if pkg == "command-line-arguments" {
		return ""
	}
	return pkg
}

// Covered returns a string describing which packages are being tested for coverage.
// If the covered package is the same as the tested package, it returns the empty string.
// Otherwise it is a comma-separated human-readable list of packages beginning with
// " in", ready for use in the coverage message.
func (t *testFuncs) Covered() string {
	if t.Cover == nil || t.Cover.Paths == nil {
		return ""
	}
	return " in " + strings.Join(t.Cover.Paths, ", ")
}

// Tested returns the name of the package being tested.
func (t *testFuncs) Tested() string {
	return t.Package.Name
}

type testFunc struct {
	Package   string // imported package name (_test or _xtest)
	Name      string // function name
	Output    string // output, for examples
	Unordered bool   // output is allowed to be unordered.
}

var testFileSet = token.NewFileSet()

func (t *testFuncs) load(filename, pkg string, doImport, seen *bool) error {
	f, err := parser.ParseFile(testFileSet, filename, nil, parser.ParseComments)
	if err != nil {
		return err
	}
	for _, d := range f.Decls {
		n, ok := d.(*ast.FuncDecl)
		if !ok {
			continue
		}
		if n.Recv != nil {
			continue
		}
		name := n.Name.String()
		switch {
		case name == "TestMain":
			if isTestFunc(n, "T") {
				t.Tests = append(t.Tests, testFunc{pkg, name, "", false})
				*doImport, *seen = true, true
				continue
			}
			err := checkTestFunc(n, "M")
			if err != nil {
				return err
			}
			if t.TestMain != nil {
				return errors.New("multiple definitions of TestMain")
			}
			t.TestMain = &testFunc{pkg, name, "", false}
			*doImport, *seen = true, true
		case isTest(name, "Test"):
			err := checkTestFunc(n, "T")
			if err != nil {
				return err
			}
			t.Tests = append(t.Tests, testFunc{pkg, name, "", false})
			*doImport, *seen = true, true
		case isTest(name, "Benchmark"):
			err := checkTestFunc(n, "B")
			if err != nil {
				return err
			}
			t.Benchmarks = append(t.Benchmarks, testFunc{pkg, name, "", false})
			*doImport, *seen = true, true
		}
	}
	ex := doc.Examples(f)
	sort.Slice(ex, func(i, j int) bool { return ex[i].Order < ex[j].Order })
	for _, e := range ex {
		*doImport = true // import test file whether executed or not
		if e.Output == "" && !e.EmptyOutput {
			// Don't run examples with no output.
			continue
		}
		t.Examples = append(t.Examples, testFunc{pkg, "Example" + e.Name, e.Output, e.Unordered})
		*seen = true
	}
	return nil
}

func checkTestFunc(fn *ast.FuncDecl, arg string) error {
	if !isTestFunc(fn, arg) {
		name := fn.Name.String()
		pos := testFileSet.Position(fn.Pos())
		return fmt.Errorf("%s: wrong signature for %s, must be: func %s(%s *testing.%s)", pos, name, name, strings.ToLower(arg), arg)
	}
	return nil
}

var testmainTmpl = lazytemplate.New("main", `
// Code generated by 'go test'. DO NOT EDIT.

package main

import (
	"os"
{{if .TestMain}}
	"reflect"
{{end}}
	"testing"
	"testing/internal/testdeps"

{{if .ImportTest}}
	{{if .NeedTest}}_test{{else}}_{{end}} {{.Package.ImportPath | printf "%q"}}
{{end}}
{{if .ImportXtest}}
	{{if .NeedXtest}}_xtest{{else}}_{{end}} {{.Package.ImportPath | printf "%s_test" | printf "%q"}}
{{end}}
{{if .Cover}}
{{range $i, $p := .Cover.Vars}}
	_cover{{$i}} {{$p.Package.ImportPath | printf "%q"}}
{{end}}
{{end}}
)

var tests = []testing.InternalTest{
{{range .Tests}}
	{"{{.Name}}", {{.Package}}.{{.Name}}},
{{end}}
}

var benchmarks = []testing.InternalBenchmark{
{{range .Benchmarks}}
	{"{{.Name}}", {{.Package}}.{{.Name}}},
{{end}}
}

var examples = []testing.InternalExample{
{{range .Examples}}
	{"{{.Name}}", {{.Package}}.{{.Name}}, {{.Output | printf "%q"}}, {{.Unordered}}},
{{end}}
}

func init() {
	testdeps.ImportPath = {{.ImportPath | printf "%q"}}
}

{{if .Cover}}

// Only updated by init functions, so no need for atomicity.
var (
	coverCounters = make(map[string][]uint32)
	coverBlocks = make(map[string][]testing.CoverBlock)
)

func init() {
	{{range $i, $p := .Cover.Vars}}
	{{range $file, $cover := $p.Vars}}
	coverRegisterFile({{printf "%q" $cover.File}}, _cover{{$i}}.{{$cover.Var}}.Count[:], _cover{{$i}}.{{$cover.Var}}.Pos[:], _cover{{$i}}.{{$cover.Var}}.NumStmt[:])
	{{end}}
	{{end}}
}

func coverRegisterFile(fileName string, counter []uint32, pos []uint32, numStmts []uint16) {
	if 3*len(counter) != len(pos) || len(counter) != len(numStmts) {
		panic("coverage: mismatched sizes")
	}
	if coverCounters[fileName] != nil {
		// Already registered.
		return
	}
	coverCounters[fileName] = counter
	block := make([]testing.CoverBlock, len(counter))
	for i := range counter {
		block[i] = testing.CoverBlock{
			Line0: pos[3*i+0],
			Col0: uint16(pos[3*i+2]),
			Line1: pos[3*i+1],
			Col1: uint16(pos[3*i+2]>>16),
			Stmts: numStmts[i],
		}
	}
	coverBlocks[fileName] = block
}
{{end}}

func main() {
{{if .Cover}}
	testing.RegisterCover(testing.Cover{
		Mode: {{printf "%q" .Cover.Mode}},
		Counters: coverCounters,
		Blocks: coverBlocks,
		CoveredPackages: {{printf "%q" .Covered}},
	})
{{end}}
	m := testing.MainStart(testdeps.TestDeps{}, tests, benchmarks, examples)
{{with .TestMain}}
	{{.Package}}.{{.Name}}(m)
	os.Exit(int(reflect.ValueOf(m).Elem().FieldByName("exitCode").Int()))
{{else}}
	os.Exit(m.Run())
{{end}}
}

`)
