// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package importer

import (
	"bytes"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"fmt"
	"go/build"
	"internal/exportdata"
	"internal/testenv"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestMain(m *testing.M) {
	build.Default.GOROOT = testenv.GOROOT(nil)
	os.Exit(m.Run())
}

// compile runs the compiler on filename, with dirname as the working directory,
// and writes the output file to outdirname.
// compile gives the resulting package a packagepath of testdata/<filebasename>.
func compile(t *testing.T, dirname, filename, outdirname string, packagefiles map[string]string) string {
	// filename must end with ".go"
	basename, ok := strings.CutSuffix(filepath.Base(filename), ".go")
	if !ok {
		t.Helper()
		t.Fatalf("filename doesn't end in .go: %s", filename)
	}
	objname := basename + ".o"
	outname := filepath.Join(outdirname, objname)
	pkgpath := path.Join("testdata", basename)

	importcfgfile := os.DevNull
	if len(packagefiles) > 0 {
		importcfgfile = filepath.Join(outdirname, basename) + ".importcfg"
		importcfg := new(bytes.Buffer)
		for k, v := range packagefiles {
			fmt.Fprintf(importcfg, "packagefile %s=%s\n", k, v)
		}
		if err := os.WriteFile(importcfgfile, importcfg.Bytes(), 0655); err != nil {
			t.Fatal(err)
		}
	}

	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "compile", "-p", pkgpath, "-D", "testdata", "-importcfg", importcfgfile, "-o", outname, filename)
	cmd.Dir = dirname
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Helper()
		t.Logf("%s", out)
		t.Fatalf("go tool compile %s failed: %s", filename, err)
	}
	return outname
}

func testPath(t *testing.T, path, srcDir string) *types2.Package {
	t0 := time.Now()
	pkg, err := Import(make(map[string]*types2.Package), path, srcDir, nil)
	if err != nil {
		t.Errorf("testPath(%s): %s", path, err)
		return nil
	}
	t.Logf("testPath(%s): %v", path, time.Since(t0))
	return pkg
}

func mktmpdir(t *testing.T) string {
	tmpdir := t.TempDir()
	if err := os.Mkdir(filepath.Join(tmpdir, "testdata"), 0700); err != nil {
		t.Fatal("mktmpdir:", err)
	}
	return tmpdir
}

func TestImportTestdata(t *testing.T) {
	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	testenv.MustHaveGoBuild(t)

	testfiles := map[string][]string{
		"exports.go":  {"go/ast"},
		"generics.go": nil,
	}

	for testfile, wantImports := range testfiles {
		tmpdir := mktmpdir(t)

		importMap := map[string]string{}
		for _, pkg := range wantImports {
			export, _, err := exportdata.FindPkg(pkg, "testdata")
			if export == "" {
				t.Fatalf("no export data found for %s: %v", pkg, err)
			}
			importMap[pkg] = export
		}

		compile(t, "testdata", testfile, filepath.Join(tmpdir, "testdata"), importMap)
		path := "./testdata/" + strings.TrimSuffix(testfile, ".go")

		if pkg := testPath(t, path, tmpdir); pkg != nil {
			// The package's Imports list must include all packages
			// explicitly imported by testfile, plus all packages
			// referenced indirectly via exported objects in testfile.
			got := fmt.Sprint(pkg.Imports())
			for _, want := range wantImports {
				if !strings.Contains(got, want) {
					t.Errorf(`Package("exports").Imports() = %s, does not contain %s`, got, want)
				}
			}
		}
	}
}

func TestVersionHandling(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	const dir = "./testdata/versions"
	list, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}

	tmpdir := mktmpdir(t)
	corruptdir := filepath.Join(tmpdir, "testdata", "versions")
	if err := os.Mkdir(corruptdir, 0700); err != nil {
		t.Fatal(err)
	}

	for _, f := range list {
		name := f.Name()
		if !strings.HasSuffix(name, ".a") {
			continue // not a package file
		}
		if strings.Contains(name, "corrupted") {
			continue // don't process a leftover corrupted file
		}
		pkgpath := "./" + name[:len(name)-2]

		if testing.Verbose() {
			t.Logf("importing %s", name)
		}

		// test that export data can be imported
		_, err := Import(make(map[string]*types2.Package), pkgpath, dir, nil)
		if err != nil {
			// ok to fail if it fails with a 'not the start of an archive file' error for select files
			if strings.Contains(err.Error(), "no longer supported") {
				switch name {
				case "test_go1.8_4.a",
					"test_go1.8_5.a":
					continue
				}
				// fall through
			}
			// ok to fail if it fails with a 'no longer supported' error for select files
			if strings.Contains(err.Error(), "no longer supported") {
				switch name {
				case "test_go1.7_0.a",
					"test_go1.7_1.a",
					"test_go1.8_4.a",
					"test_go1.8_5.a",
					"test_go1.11_6b.a",
					"test_go1.11_999b.a":
					continue
				}
				// fall through
			}
			// ok to fail if it fails with a 'newer version' error for select files
			if strings.Contains(err.Error(), "newer version") {
				switch name {
				case "test_go1.11_999i.a":
					continue
				}
				// fall through
			}
			t.Errorf("import %q failed: %v", pkgpath, err)
			continue
		}

		// create file with corrupted export data
		// 1) read file
		data, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatal(err)
		}
		// 2) find export data
		i := bytes.Index(data, []byte("\n$$B\n")) + 5
		// Export data can contain "\n$$\n" in string constants, however,
		// searching for the next end of section marker "\n$$\n" is good enough for testzs.
		j := bytes.Index(data[i:], []byte("\n$$\n")) + i
		if i < 0 || j < 0 || i > j {
			t.Fatalf("export data section not found (i = %d, j = %d)", i, j)
		}
		// 3) corrupt the data (increment every 7th byte)
		for k := j - 13; k >= i; k -= 7 {
			data[k]++
		}
		// 4) write the file
		pkgpath += "_corrupted"
		filename := filepath.Join(corruptdir, pkgpath) + ".a"
		os.WriteFile(filename, data, 0666)

		// test that importing the corrupted file results in an error
		_, err = Import(make(map[string]*types2.Package), pkgpath, corruptdir, nil)
		if err == nil {
			t.Errorf("import corrupted %q succeeded", pkgpath)
		} else if msg := err.Error(); !strings.Contains(msg, "version skew") {
			t.Errorf("import %q error incorrect (%s)", pkgpath, msg)
		}
	}
}

func TestImportStdLib(t *testing.T) {
	if testing.Short() {
		t.Skip("the imports can be expensive, and this test is especially slow when the build cache is empty")
	}
	testenv.MustHaveGoBuild(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	// Get list of packages in stdlib. Filter out test-only packages with {{if .GoFiles}} check.
	var stderr bytes.Buffer
	cmd := exec.Command(testenv.GoToolPath(t), "list", "-f", "{{if .GoFiles}}{{.ImportPath}}{{end}}", "std")
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("failed to run go list to determine stdlib packages: %v\nstderr:\n%v", err, stderr.String())
	}
	pkgs := strings.Fields(string(out))

	var nimports int
	for _, pkg := range pkgs {
		t.Run(pkg, func(t *testing.T) {
			if testPath(t, pkg, filepath.Join(testenv.GOROOT(t), "src", path.Dir(pkg))) != nil {
				nimports++
			}
		})
	}
	const minPkgs = 225 // 'GOOS=plan9 go1.18 list std | wc -l' reports 228; most other platforms have more.
	if len(pkgs) < minPkgs {
		t.Fatalf("too few packages (%d) were imported", nimports)
	}

	t.Logf("tested %d imports", nimports)
}

var importedObjectTests = []struct {
	name string
	want string
}{
	// non-interfaces
	{"crypto.Hash", "type Hash uint"},
	{"go/ast.ObjKind", "type ObjKind int"},
	{"go/types.Qualifier", "type Qualifier func(*Package) string"},
	{"go/types.Comparable", "func Comparable(T Type) bool"},
	{"math.Pi", "const Pi untyped float"},
	{"math.Sin", "func Sin(x float64) float64"},
	{"go/ast.NotNilFilter", "func NotNilFilter(_ string, v reflect.Value) bool"},
	{"internal/exportdata.FindPkg", "func FindPkg(path string, srcDir string) (filename string, id string, err error)"},

	// interfaces
	{"context.Context", "type Context interface{Deadline() (deadline time.Time, ok bool); Done() <-chan struct{}; Err() error; Value(key any) any}"},
	{"crypto.Decrypter", "type Decrypter interface{Decrypt(rand io.Reader, msg []byte, opts DecrypterOpts) (plaintext []byte, err error); Public() PublicKey}"},
	{"encoding.BinaryMarshaler", "type BinaryMarshaler interface{MarshalBinary() (data []byte, err error)}"},
	{"io.Reader", "type Reader interface{Read(p []byte) (n int, err error)}"},
	{"io.ReadWriter", "type ReadWriter interface{Reader; Writer}"},
	{"go/ast.Node", "type Node interface{End() go/token.Pos; Pos() go/token.Pos}"},
	{"go/types.Type", "type Type interface{String() string; Underlying() Type}"},
}

func TestImportedTypes(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	for _, test := range importedObjectTests {
		s := strings.Split(test.name, ".")
		if len(s) != 2 {
			t.Fatal("inconsistent test data")
		}
		importPath := s[0]
		objName := s[1]

		pkg, err := Import(make(map[string]*types2.Package), importPath, ".", nil)
		if err != nil {
			t.Error(err)
			continue
		}

		obj := pkg.Scope().Lookup(objName)
		if obj == nil {
			t.Errorf("%s: object not found", test.name)
			continue
		}

		got := types2.ObjectString(obj, types2.RelativeTo(pkg))
		if got != test.want {
			t.Errorf("%s: got %q; want %q", test.name, got, test.want)
		}

		if named, _ := obj.Type().(*types2.Named); named != nil {
			verifyInterfaceMethodRecvs(t, named, 0)
		}
	}
}

// verifyInterfaceMethodRecvs verifies that method receiver types
// are named if the methods belong to a named interface type.
func verifyInterfaceMethodRecvs(t *testing.T, named *types2.Named, level int) {
	// avoid endless recursion in case of an embedding bug that lead to a cycle
	if level > 10 {
		t.Errorf("%s: embeds itself", named)
		return
	}

	iface, _ := named.Underlying().(*types2.Interface)
	if iface == nil {
		return // not an interface
	}

	// The unified IR importer always sets interface method receiver
	// parameters to point to the Interface type, rather than the Named.
	// See #49906.
	var want types2.Type = iface

	// check explicitly declared methods
	for i := 0; i < iface.NumExplicitMethods(); i++ {
		m := iface.ExplicitMethod(i)
		recv := m.Type().(*types2.Signature).Recv()
		if recv == nil {
			t.Errorf("%s: missing receiver type", m)
			continue
		}
		if recv.Type() != want {
			t.Errorf("%s: got recv type %s; want %s", m, recv.Type(), named)
		}
	}

	// check embedded interfaces (if they are named, too)
	for i := 0; i < iface.NumEmbeddeds(); i++ {
		// embedding of interfaces cannot have cycles; recursion will terminate
		if etype, _ := iface.EmbeddedType(i).(*types2.Named); etype != nil {
			verifyInterfaceMethodRecvs(t, etype, level+1)
		}
	}
}

func TestIssue5815(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	pkg := importPkg(t, "strings", ".")

	scope := pkg.Scope()
	for _, name := range scope.Names() {
		obj := scope.Lookup(name)
		if obj.Pkg() == nil {
			t.Errorf("no pkg for %s", obj)
		}
		if tname, _ := obj.(*types2.TypeName); tname != nil {
			named := tname.Type().(*types2.Named)
			for i := 0; i < named.NumMethods(); i++ {
				m := named.Method(i)
				if m.Pkg() == nil {
					t.Errorf("no pkg for %s", m)
				}
			}
		}
	}
}

// Smoke test to ensure that imported methods get the correct package.
func TestCorrectMethodPackage(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	imports := make(map[string]*types2.Package)
	_, err := Import(imports, "net/http", ".", nil)
	if err != nil {
		t.Fatal(err)
	}

	mutex := imports["sync"].Scope().Lookup("Mutex").(*types2.TypeName).Type()
	obj, _, _ := types2.LookupFieldOrMethod(types2.NewPointer(mutex), false, nil, "Lock")
	lock := obj.(*types2.Func)
	if got, want := lock.Pkg().Path(), "sync"; got != want {
		t.Errorf("got package path %q; want %q", got, want)
	}
}

func TestIssue13566(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	tmpdir := mktmpdir(t)
	testoutdir := filepath.Join(tmpdir, "testdata")

	// b.go needs to be compiled from the output directory so that the compiler can
	// find the compiled package a. We pass the full path to compile() so that we
	// don't have to copy the file to that directory.
	bpath, err := filepath.Abs(filepath.Join("testdata", "b.go"))
	if err != nil {
		t.Fatal(err)
	}

	jsonExport, _, err := exportdata.FindPkg("encoding/json", "testdata")
	if jsonExport == "" {
		t.Fatalf("no export data found for encoding/json: %v", err)
	}

	compile(t, "testdata", "a.go", testoutdir, map[string]string{"encoding/json": jsonExport})
	compile(t, testoutdir, bpath, testoutdir, map[string]string{"testdata/a": filepath.Join(testoutdir, "a.o")})

	// import must succeed (test for issue at hand)
	pkg := importPkg(t, "./testdata/b", tmpdir)

	// make sure all indirectly imported packages have names
	for _, imp := range pkg.Imports() {
		if imp.Name() == "" {
			t.Errorf("no name for %s package", imp.Path())
		}
	}
}

func TestIssue13898(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	// import go/internal/gcimporter which imports go/types partially
	imports := make(map[string]*types2.Package)
	_, err := Import(imports, "go/internal/gcimporter", ".", nil)
	if err != nil {
		t.Fatal(err)
	}

	// look for go/types package
	var goTypesPkg *types2.Package
	for path, pkg := range imports {
		if path == "go/types" {
			goTypesPkg = pkg
			break
		}
	}
	if goTypesPkg == nil {
		t.Fatal("go/types not found")
	}

	// look for go/types.Object type
	obj := lookupObj(t, goTypesPkg.Scope(), "Object")
	typ, ok := obj.Type().(*types2.Named)
	if !ok {
		t.Fatalf("go/types.Object type is %v; wanted named type", typ)
	}

	// lookup go/types.Object.Pkg method
	m, index, indirect := types2.LookupFieldOrMethod(typ, false, nil, "Pkg")
	if m == nil {
		t.Fatalf("go/types.Object.Pkg not found (index = %v, indirect = %v)", index, indirect)
	}

	// the method must belong to go/types
	if m.Pkg().Path() != "go/types" {
		t.Fatalf("found %v; want go/types", m.Pkg())
	}
}

func TestIssue15517(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	tmpdir := mktmpdir(t)

	compile(t, "testdata", "p.go", filepath.Join(tmpdir, "testdata"), nil)

	// Multiple imports of p must succeed without redeclaration errors.
	// We use an import path that's not cleaned up so that the eventual
	// file path for the package is different from the package path; this
	// will expose the error if it is present.
	//
	// (Issue: Both the textual and the binary importer used the file path
	// of the package to be imported as key into the shared packages map.
	// However, the binary importer then used the package path to identify
	// the imported package to mark it as complete; effectively marking the
	// wrong package as complete. By using an "unclean" package path, the
	// file and package path are different, exposing the problem if present.
	// The same issue occurs with vendoring.)
	imports := make(map[string]*types2.Package)
	for i := 0; i < 3; i++ {
		if _, err := Import(imports, "./././testdata/p", tmpdir, nil); err != nil {
			t.Fatal(err)
		}
	}
}

func TestIssue15920(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	compileAndImportPkg(t, "issue15920")
}

func TestIssue20046(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	// "./issue20046".V.M must exist
	pkg := compileAndImportPkg(t, "issue20046")
	obj := lookupObj(t, pkg.Scope(), "V")
	if m, index, indirect := types2.LookupFieldOrMethod(obj.Type(), false, nil, "M"); m == nil {
		t.Fatalf("V.M not found (index = %v, indirect = %v)", index, indirect)
	}
}
func TestIssue25301(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	compileAndImportPkg(t, "issue25301")
}

func TestIssue25596(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	compileAndImportPkg(t, "issue25596")
}

func importPkg(t *testing.T, path, srcDir string) *types2.Package {
	pkg, err := Import(make(map[string]*types2.Package), path, srcDir, nil)
	if err != nil {
		t.Helper()
		t.Fatal(err)
	}
	return pkg
}

func compileAndImportPkg(t *testing.T, name string) *types2.Package {
	t.Helper()
	tmpdir := mktmpdir(t)
	compile(t, "testdata", name+".go", filepath.Join(tmpdir, "testdata"), nil)
	return importPkg(t, "./testdata/"+name, tmpdir)
}

func lookupObj(t *testing.T, scope *types2.Scope, name string) types2.Object {
	if obj := scope.Lookup(name); obj != nil {
		return obj
	}
	t.Helper()
	t.Fatalf("%s not found", name)
	return nil
}

// importMap implements the types2.Importer interface.
type importMap map[string]*types2.Package

func (m importMap) Import(path string) (*types2.Package, error) { return m[path], nil }

func TestIssue69912(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	tmpdir := t.TempDir()
	testoutdir := filepath.Join(tmpdir, "testdata")
	if err := os.Mkdir(testoutdir, 0700); err != nil {
		t.Fatalf("making output dir: %v", err)
	}

	compile(t, "testdata", "issue69912.go", testoutdir, nil)

	issue69912, err := Import(make(map[string]*types2.Package), "./testdata/issue69912", tmpdir, nil)
	if err != nil {
		t.Fatal(err)
	}

	check := func(pkgname, src string, imports importMap) (*types2.Package, error) {
		f, err := syntax.Parse(syntax.NewFileBase(pkgname), strings.NewReader(src), nil, nil, 0)
		if err != nil {
			return nil, err
		}
		config := &types2.Config{
			Importer: imports,
		}
		return config.Check(pkgname, []*syntax.File{f}, nil)
	}

	// Use the resulting package concurrently, via dot-imports, to exercise the
	// race of issue #69912.
	const pSrc = `package p

import . "issue69912"

type S struct {
	f T
}
`
	importer := importMap{
		"issue69912": issue69912,
	}
	var wg sync.WaitGroup
	for range 10 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := check("p", pSrc, importer); err != nil {
				t.Errorf("Check failed: %v", err)
			}
		}()
	}
	wg.Wait()
}
