// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcimporter_test

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"go/ast"
	"go/build"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"

	. "go/internal/gcimporter"
)

func TestMain(m *testing.M) {
	build.Default.GOROOT = testenv.GOROOT(nil)
	os.Exit(m.Run())
}

// compile runs the compiler on filename, with dirname as the working directory,
// and writes the output file to outdirname.
// compile gives the resulting package a packagepath of testdata/<filebasename>.
func compile(t *testing.T, dirname, filename, outdirname string, packageFiles map[string]string, pkgImports ...string) string {
	// filename must end with ".go"
	basename, ok := strings.CutSuffix(filepath.Base(filename), ".go")
	if !ok {
		t.Fatalf("filename doesn't end in .go: %s", filename)
	}
	objname := basename + ".o"
	outname := filepath.Join(outdirname, objname)

	importcfgfile := os.DevNull
	if len(packageFiles) > 0 || len(pkgImports) > 0 {
		importcfgfile = filepath.Join(outdirname, basename) + ".importcfg"
		testenv.WriteImportcfg(t, importcfgfile, packageFiles, pkgImports...)
	}

	pkgpath := path.Join("testdata", basename)
	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "compile", "-p", pkgpath, "-D", "testdata", "-importcfg", importcfgfile, "-o", outname, filename)
	cmd.Dir = dirname
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Logf("%s", out)
		t.Fatalf("go tool compile %s failed: %s", filename, err)
	}
	return outname
}

func testPath(t *testing.T, path, srcDir string) *types.Package {
	t0 := time.Now()
	fset := token.NewFileSet()
	pkg, err := Import(fset, make(map[string]*types.Package), path, srcDir, nil)
	if err != nil {
		t.Errorf("testPath(%s): %s", path, err)
		return nil
	}
	t.Logf("testPath(%s): %v", path, time.Since(t0))
	return pkg
}

var pkgExts = [...]string{".a", ".o"} // keep in sync with gcimporter.go

func mktmpdir(t *testing.T) string {
	tmpdir, err := os.MkdirTemp("", "gcimporter_test")
	if err != nil {
		t.Fatal("mktmpdir:", err)
	}
	if err := os.Mkdir(filepath.Join(tmpdir, "testdata"), 0700); err != nil {
		os.RemoveAll(tmpdir)
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
		"exports.go":  {"go/ast", "go/token"},
		"generics.go": nil,
	}
	if true /* was goexperiment.Unified */ {
		// TODO(mdempsky): Fix test below to flatten the transitive
		// Package.Imports graph. Unified IR is more precise about
		// recreating the package import graph.
		testfiles["exports.go"] = []string{"go/ast"}
	}

	for testfile, wantImports := range testfiles {
		tmpdir := mktmpdir(t)
		defer os.RemoveAll(tmpdir)

		compile(t, "testdata", testfile, filepath.Join(tmpdir, "testdata"), nil, wantImports...)
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

func TestImportTypeparamTests(t *testing.T) {
	if testing.Short() {
		t.Skipf("in short mode, skipping test that requires export data for all of std")
	}

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	// cmd/distpack removes the GOROOT/test directory, so skip if it isn't there.
	// cmd/distpack also requires the presence of GOROOT/VERSION, so use that to
	// avoid false-positive skips.
	gorootTest := filepath.Join(testenv.GOROOT(t), "test")
	if _, err := os.Stat(gorootTest); os.IsNotExist(err) {
		if _, err := os.Stat(filepath.Join(testenv.GOROOT(t), "VERSION")); err == nil {
			t.Skipf("skipping: GOROOT/test not present")
		}
	}

	testenv.MustHaveGoBuild(t)

	tmpdir := mktmpdir(t)
	defer os.RemoveAll(tmpdir)

	// Check go files in test/typeparam, except those that fail for a known
	// reason.
	rootDir := filepath.Join(gorootTest, "typeparam")
	list, err := os.ReadDir(rootDir)
	if err != nil {
		t.Fatal(err)
	}

	for _, entry := range list {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") {
			// For now, only consider standalone go files.
			continue
		}

		t.Run(entry.Name(), func { t ->
			filename := filepath.Join(rootDir, entry.Name())
			src, err := os.ReadFile(filename)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.HasPrefix(src, []byte("// run")) && !bytes.HasPrefix(src, []byte("// compile")) {
				// We're bypassing the logic of run.go here, so be conservative about
				// the files we consider in an attempt to make this test more robust to
				// changes in test/typeparams.
				t.Skipf("not detected as a run test")
			}

			// Compile and import, and compare the resulting package with the package
			// that was type-checked directly.
			compile(t, rootDir, entry.Name(), filepath.Join(tmpdir, "testdata"), nil, filename)
			pkgName := strings.TrimSuffix(entry.Name(), ".go")
			imported := importPkg(t, "./testdata/"+pkgName, tmpdir)
			checked := checkFile(t, filename, src)

			seen := make(map[string]bool)
			for _, name := range imported.Scope().Names() {
				if !token.IsExported(name) {
					continue // ignore synthetic names like .inittask and .dict.*
				}
				seen[name] = true

				importedObj := imported.Scope().Lookup(name)
				got := types.ObjectString(importedObj, types.RelativeTo(imported))
				got = sanitizeObjectString(got)

				checkedObj := checked.Scope().Lookup(name)
				if checkedObj == nil {
					t.Fatalf("imported object %q was not type-checked", name)
				}
				want := types.ObjectString(checkedObj, types.RelativeTo(checked))
				want = sanitizeObjectString(want)

				if got != want {
					t.Errorf("imported %q as %q, want %q", name, got, want)
				}
			}

			for _, name := range checked.Scope().Names() {
				if !token.IsExported(name) || seen[name] {
					continue
				}
				t.Errorf("did not import object %q", name)
			}
		})
	}
}

// sanitizeObjectString removes type parameter debugging markers from an object
// string, to normalize it for comparison.
// TODO(rfindley): this should not be necessary.
func sanitizeObjectString(s string) string {
	var runes []rune
	for _, r := range s {
		if '₀' <= r && r < '₀'+10 {
			continue // trim type parameter subscripts
		}
		runes = append(runes, r)
	}
	return string(runes)
}

func checkFile(t *testing.T, filename string, src []byte) *types.Package {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, filename, src, 0)
	if err != nil {
		t.Fatal(err)
	}
	config := types.Config{
		Importer: importer.Default(),
	}
	pkg, err := config.Check("", fset, []*ast.File{f}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return pkg
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
	defer os.RemoveAll(tmpdir)
	corruptdir := filepath.Join(tmpdir, "testdata", "versions")
	if err := os.Mkdir(corruptdir, 0700); err != nil {
		t.Fatal(err)
	}

	fset := token.NewFileSet()

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
		_, err := Import(fset, make(map[string]*types.Package), pkgpath, dir, nil)
		if err != nil {
			// ok to fail if it fails with a no longer supported error for select files
			if strings.Contains(err.Error(), "no longer supported") {
				switch name {
				case "test_go1.7_0.a", "test_go1.7_1.a",
					"test_go1.8_4.a", "test_go1.8_5.a",
					"test_go1.11_6b.a", "test_go1.11_999b.a":
					continue
				}
				// fall through
			}
			// ok to fail if it fails with a newer version error for select files
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
		_, err = Import(fset, make(map[string]*types.Package), pkgpath, corruptdir, nil)
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
	cmd := exec.Command("go", "list", "-f", "{{if .GoFiles}}{{.ImportPath}}{{end}}", "std")
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("failed to run go list to determine stdlib packages: %v\nstderr:\n%v", err, stderr.String())
	}
	pkgs := strings.Fields(string(out))

	var nimports int
	for _, pkg := range pkgs {
		t.Run(pkg, func { t ->
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
	{"go/internal/gcimporter.FindPkg", "func FindPkg(path string, srcDir string) (filename string, id string, err error)"},

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

	fset := token.NewFileSet()
	for _, test := range importedObjectTests {
		s := strings.Split(test.name, ".")
		if len(s) != 2 {
			t.Fatal("inconsistent test data")
		}
		importPath := s[0]
		objName := s[1]

		pkg, err := Import(fset, make(map[string]*types.Package), importPath, ".", nil)
		if err != nil {
			t.Error(err)
			continue
		}

		obj := pkg.Scope().Lookup(objName)
		if obj == nil {
			t.Errorf("%s: object not found", test.name)
			continue
		}

		got := types.ObjectString(obj, types.RelativeTo(pkg))
		if got != test.want {
			t.Errorf("%s: got %q; want %q", test.name, got, test.want)
		}

		if named, _ := obj.Type().(*types.Named); named != nil {
			verifyInterfaceMethodRecvs(t, named, 0)
		}
	}
}

// verifyInterfaceMethodRecvs verifies that method receiver types
// are named if the methods belong to a named interface type.
func verifyInterfaceMethodRecvs(t *testing.T, named *types.Named, level int) {
	// avoid endless recursion in case of an embedding bug that lead to a cycle
	if level > 10 {
		t.Errorf("%s: embeds itself", named)
		return
	}

	iface, _ := named.Underlying().(*types.Interface)
	if iface == nil {
		return // not an interface
	}

	// check explicitly declared methods
	for i := 0; i < iface.NumExplicitMethods(); i++ {
		m := iface.ExplicitMethod(i)
		recv := m.Type().(*types.Signature).Recv()
		if recv == nil {
			t.Errorf("%s: missing receiver type", m)
			continue
		}
		if recv.Type() != named {
			t.Errorf("%s: got recv type %s; want %s", m, recv.Type(), named)
		}
	}

	// check embedded interfaces (if they are named, too)
	for i := 0; i < iface.NumEmbeddeds(); i++ {
		// embedding of interfaces cannot have cycles; recursion will terminate
		if etype, _ := iface.EmbeddedType(i).(*types.Named); etype != nil {
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
		if tname, _ := obj.(*types.TypeName); tname != nil {
			named := tname.Type().(*types.Named)
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

	imports := make(map[string]*types.Package)
	fset := token.NewFileSet()
	_, err := Import(fset, imports, "net/http", ".", nil)
	if err != nil {
		t.Fatal(err)
	}

	mutex := imports["sync"].Scope().Lookup("Mutex").(*types.TypeName).Type()
	mset := types.NewMethodSet(types.NewPointer(mutex)) // methods of *sync.Mutex
	sel := mset.Lookup(nil, "Lock")
	lock := sel.Obj().(*types.Func)
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
	defer os.RemoveAll(tmpdir)
	testoutdir := filepath.Join(tmpdir, "testdata")

	// b.go needs to be compiled from the output directory so that the compiler can
	// find the compiled package a. We pass the full path to compile() so that we
	// don't have to copy the file to that directory.
	bpath, err := filepath.Abs(filepath.Join("testdata", "b.go"))
	if err != nil {
		t.Fatal(err)
	}

	compile(t, "testdata", "a.go", testoutdir, nil, "encoding/json")
	compile(t, testoutdir, bpath, testoutdir, map[string]string{"testdata/a": filepath.Join(testoutdir, "a.o")}, "encoding/json")

	// import must succeed (test for issue at hand)
	pkg := importPkg(t, "./testdata/b", tmpdir)

	// make sure all indirectly imported packages have names
	for _, imp := range pkg.Imports() {
		if imp.Name() == "" {
			t.Errorf("no name for %s package", imp.Path())
		}
	}
}

func TestTypeNamingOrder(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	tmpdir := mktmpdir(t)
	defer os.RemoveAll(tmpdir)
	testoutdir := filepath.Join(tmpdir, "testdata")

	compile(t, "testdata", "g.go", testoutdir, nil)

	// import must succeed (test for issue at hand)
	_ = importPkg(t, "./testdata/g", tmpdir)
}

func TestIssue13898(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	// import go/internal/gcimporter which imports go/types partially
	fset := token.NewFileSet()
	imports := make(map[string]*types.Package)
	_, err := Import(fset, imports, "go/internal/gcimporter", ".", nil)
	if err != nil {
		t.Fatal(err)
	}

	// look for go/types package
	var goTypesPkg *types.Package
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
	typ, ok := obj.Type().(*types.Named)
	if !ok {
		t.Fatalf("go/types.Object type is %v; wanted named type", typ)
	}

	// lookup go/types.Object.Pkg method
	m, index, indirect := types.LookupFieldOrMethod(typ, false, nil, "Pkg")
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
	defer os.RemoveAll(tmpdir)

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
	imports := make(map[string]*types.Package)
	fset := token.NewFileSet()
	for i := 0; i < 3; i++ {
		if _, err := Import(fset, imports, "./././testdata/p", tmpdir, nil); err != nil {
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
	if m, index, indirect := types.LookupFieldOrMethod(obj.Type(), false, nil, "M"); m == nil {
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

func TestIssue57015(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	compileAndImportPkg(t, "issue57015")
}

func importPkg(t *testing.T, path, srcDir string) *types.Package {
	fset := token.NewFileSet()
	pkg, err := Import(fset, make(map[string]*types.Package), path, srcDir, nil)
	if err != nil {
		t.Helper()
		t.Fatal(err)
	}
	return pkg
}

func compileAndImportPkg(t *testing.T, name string) *types.Package {
	t.Helper()
	tmpdir := mktmpdir(t)
	defer os.RemoveAll(tmpdir)
	compile(t, "testdata", name+".go", filepath.Join(tmpdir, "testdata"), nil)
	return importPkg(t, "./testdata/"+name, tmpdir)
}

func lookupObj(t *testing.T, scope *types.Scope, name string) types.Object {
	if obj := scope.Lookup(name); obj != nil {
		return obj
	}
	t.Helper()
	t.Fatalf("%s not found", name)
	return nil
}
