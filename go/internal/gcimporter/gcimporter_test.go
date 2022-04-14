// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is a copy of $GOROOT/src/go/internal/gcimporter/gcimporter_test.go,
// adjusted to make it build with code from (std lib) internal/testenv copied.

package gcimporter

import (
	"bytes"
	"fmt"
	"go/build"
	"go/constant"
	"go/types"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"golang.org/x/tools/internal/testenv"
)

func TestMain(m *testing.M) {
	testenv.ExitIfSmallMachine()
	os.Exit(m.Run())
}

// ----------------------------------------------------------------------------

func needsCompiler(t *testing.T, compiler string) {
	if runtime.Compiler == compiler {
		return
	}
	switch compiler {
	case "gc":
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}
}

// compile runs the compiler on filename, with dirname as the working directory,
// and writes the output file to outdirname.
func compile(t *testing.T, dirname, filename, outdirname string) string {
	testenv.NeedsGoBuild(t)

	// filename must end with ".go"
	if !strings.HasSuffix(filename, ".go") {
		t.Fatalf("filename doesn't end in .go: %s", filename)
	}
	basename := filepath.Base(filename)
	outname := filepath.Join(outdirname, basename[:len(basename)-2]+"o")
	cmd := exec.Command("go", "tool", "compile", "-p=p", "-o", outname, filename)
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
	pkg, err := Import(make(map[string]*types.Package), path, srcDir, nil)
	if err != nil {
		t.Errorf("testPath(%s): %s", path, err)
		return nil
	}
	t.Logf("testPath(%s): %v", path, time.Since(t0))
	return pkg
}

const maxTime = 30 * time.Second

func testDir(t *testing.T, dir string, endTime time.Time) (nimports int) {
	dirname := filepath.Join(runtime.GOROOT(), "pkg", runtime.GOOS+"_"+runtime.GOARCH, dir)
	list, err := ioutil.ReadDir(dirname)
	if err != nil {
		t.Fatalf("testDir(%s): %s", dirname, err)
	}
	for _, f := range list {
		if time.Now().After(endTime) {
			t.Log("testing time used up")
			return
		}
		switch {
		case !f.IsDir():
			// try extensions
			for _, ext := range pkgExts {
				if strings.HasSuffix(f.Name(), ext) {
					name := f.Name()[0 : len(f.Name())-len(ext)] // remove extension
					if testPath(t, filepath.Join(dir, name), dir) != nil {
						nimports++
					}
				}
			}
		case f.IsDir():
			nimports += testDir(t, filepath.Join(dir, f.Name()), endTime)
		}
	}
	return
}

func mktmpdir(t *testing.T) string {
	tmpdir, err := ioutil.TempDir("", "gcimporter_test")
	if err != nil {
		t.Fatal("mktmpdir:", err)
	}
	if err := os.Mkdir(filepath.Join(tmpdir, "testdata"), 0700); err != nil {
		os.RemoveAll(tmpdir)
		t.Fatal("mktmpdir:", err)
	}
	return tmpdir
}

const testfile = "exports.go"

func TestImportTestdata(t *testing.T) {
	needsCompiler(t, "gc")

	tmpdir := mktmpdir(t)
	defer os.RemoveAll(tmpdir)

	compile(t, "testdata", testfile, filepath.Join(tmpdir, "testdata"))

	// filename should end with ".go"
	filename := testfile[:len(testfile)-3]
	if pkg := testPath(t, "./testdata/"+filename, tmpdir); pkg != nil {
		// The package's Imports list must include all packages
		// explicitly imported by testfile, plus all packages
		// referenced indirectly via exported objects in testfile.
		// With the textual export format (when run against Go1.6),
		// the list may also include additional packages that are
		// not strictly required for import processing alone (they
		// are exported to err "on the safe side").
		// For now, we just test the presence of a few packages
		// that we know are there for sure.
		got := fmt.Sprint(pkg.Imports())
		for _, want := range []string{"go/ast", "go/token"} {
			if !strings.Contains(got, want) {
				t.Errorf(`Package("exports").Imports() = %s, does not contain %s`, got, want)
			}
		}
	}
}

func TestVersionHandling(t *testing.T) {
	if debug {
		t.Skip("TestVersionHandling panics in debug mode")
	}

	// This package only handles gc export data.
	needsCompiler(t, "gc")

	const dir = "./testdata/versions"
	list, err := ioutil.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}

	tmpdir := mktmpdir(t)
	defer os.RemoveAll(tmpdir)
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
		_, err := Import(make(map[string]*types.Package), pkgpath, dir, nil)
		if err != nil {
			// ok to fail if it fails with a newer version error for select files
			if strings.Contains(err.Error(), "newer version") {
				switch name {
				case "test_go1.11_999b.a", "test_go1.11_999i.a":
					continue
				}
				// fall through
			}
			t.Errorf("import %q failed: %v", pkgpath, err)
			continue
		}

		// create file with corrupted export data
		// 1) read file
		data, err := ioutil.ReadFile(filepath.Join(dir, name))
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
		ioutil.WriteFile(filename, data, 0666)

		// test that importing the corrupted file results in an error
		_, err = Import(make(map[string]*types.Package), pkgpath, corruptdir, nil)
		if err == nil {
			t.Errorf("import corrupted %q succeeded", pkgpath)
		} else if msg := err.Error(); !strings.Contains(msg, "version skew") {
			t.Errorf("import %q error incorrect (%s)", pkgpath, msg)
		}
	}
}

func TestImportStdLib(t *testing.T) {
	// This package only handles gc export data.
	needsCompiler(t, "gc")

	dt := maxTime
	if testing.Short() && os.Getenv("GO_BUILDER_NAME") == "" {
		dt = 10 * time.Millisecond
	}
	nimports := testDir(t, "", time.Now().Add(dt)) // installed packages
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
	{"go/internal/gcimporter.FindPkg", "func FindPkg(path string, srcDir string) (filename string, id string)"},

	// interfaces
	{"context.Context", "type Context interface{Deadline() (deadline time.Time, ok bool); Done() <-chan struct{}; Err() error; Value(key any) any}"},
	{"crypto.Decrypter", "type Decrypter interface{Decrypt(rand io.Reader, msg []byte, opts DecrypterOpts) (plaintext []byte, err error); Public() PublicKey}"},
	{"encoding.BinaryMarshaler", "type BinaryMarshaler interface{MarshalBinary() (data []byte, err error)}"},
	{"io.Reader", "type Reader interface{Read(p []byte) (n int, err error)}"},
	{"io.ReadWriter", "type ReadWriter interface{Reader; Writer}"},
	{"go/ast.Node", "type Node interface{End() go/token.Pos; Pos() go/token.Pos}"},
	{"go/types.Type", "type Type interface{String() string; Underlying() Type}"},
}

// TODO(rsc): Delete this init func after x/tools no longer needs to test successfully with Go 1.17.
func init() {
	if build.Default.ReleaseTags[len(build.Default.ReleaseTags)-1] <= "go1.17" {
		for i := range importedObjectTests {
			if importedObjectTests[i].name == "context.Context" {
				// Expand any to interface{}.
				importedObjectTests[i].want = "type Context interface{Deadline() (deadline time.Time, ok bool); Done() <-chan struct{}; Err() error; Value(key interface{}) interface{}}"
			}
		}
	}
}

func TestImportedTypes(t *testing.T) {
	testenv.NeedsGo1Point(t, 11)
	// This package only handles gc export data.
	needsCompiler(t, "gc")

	for _, test := range importedObjectTests {
		obj := importObject(t, test.name)
		if obj == nil {
			continue // error reported elsewhere
		}
		got := types.ObjectString(obj, types.RelativeTo(obj.Pkg()))

		// TODO(rsc): Delete this block once go.dev/cl/368254 lands.
		if got != test.want && test.want == strings.ReplaceAll(got, "interface{}", "any") {
			got = test.want
		}

		if got != test.want {
			t.Errorf("%s: got %q; want %q", test.name, got, test.want)
		}

		if named, _ := obj.Type().(*types.Named); named != nil {
			verifyInterfaceMethodRecvs(t, named, 0)
		}
	}
}

func TestImportedConsts(t *testing.T) {
	testenv.NeedsGo1Point(t, 11)
	tests := []struct {
		name string
		want constant.Kind
	}{
		{"math.Pi", constant.Float},
		{"math.MaxFloat64", constant.Float},
		{"math.MaxInt64", constant.Int},
	}

	for _, test := range tests {
		obj := importObject(t, test.name)
		if got := obj.(*types.Const).Val().Kind(); got != test.want {
			t.Errorf("%s: imported as constant.Kind(%v), want constant.Kind(%v)", test.name, got, test.want)
		}
	}
}

// importObject imports the object specified by a name of the form
// <import path>.<object name>, e.g. go/types.Type.
//
// If any errors occur they are reported via t and the resulting object will
// be nil.
func importObject(t *testing.T, name string) types.Object {
	s := strings.Split(name, ".")
	if len(s) != 2 {
		t.Fatal("inconsistent test data")
	}
	importPath := s[0]
	objName := s[1]

	pkg, err := Import(make(map[string]*types.Package), importPath, ".", nil)
	if err != nil {
		t.Error(err)
		return nil
	}

	obj := pkg.Scope().Lookup(objName)
	if obj == nil {
		t.Errorf("%s: object not found", name)
		return nil
	}
	return obj
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
	// This package only handles gc export data.
	needsCompiler(t, "gc")

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
	// This package only handles gc export data.
	needsCompiler(t, "gc")

	imports := make(map[string]*types.Package)
	_, err := Import(imports, "net/http", ".", nil)
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
	// This package only handles gc export data.
	needsCompiler(t, "gc")

	// On windows, we have to set the -D option for the compiler to avoid having a drive
	// letter and an illegal ':' in the import path - just skip it (see also issue #3483).
	if runtime.GOOS == "windows" {
		t.Skip("avoid dealing with relative paths/drive letters on windows")
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
	compile(t, "testdata", "a.go", testoutdir)
	compile(t, testoutdir, bpath, testoutdir)

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
	// This package only handles gc export data.
	needsCompiler(t, "gc")

	// import go/internal/gcimporter which imports go/types partially
	imports := make(map[string]*types.Package)
	_, err := Import(imports, "go/internal/gcimporter", ".", nil)
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
	// This package only handles gc export data.
	needsCompiler(t, "gc")

	// On windows, we have to set the -D option for the compiler to avoid having a drive
	// letter and an illegal ':' in the import path - just skip it (see also issue #3483).
	if runtime.GOOS == "windows" {
		t.Skip("avoid dealing with relative paths/drive letters on windows")
	}

	tmpdir := mktmpdir(t)
	defer os.RemoveAll(tmpdir)

	compile(t, "testdata", "p.go", filepath.Join(tmpdir, "testdata"))

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
	for i := 0; i < 3; i++ {
		if _, err := Import(imports, "./././testdata/p", tmpdir, nil); err != nil {
			t.Fatal(err)
		}
	}
}

func TestIssue15920(t *testing.T) {
	// This package only handles gc export data.
	needsCompiler(t, "gc")

	// On windows, we have to set the -D option for the compiler to avoid having a drive
	// letter and an illegal ':' in the import path - just skip it (see also issue #3483).
	if runtime.GOOS == "windows" {
		t.Skip("avoid dealing with relative paths/drive letters on windows")
	}

	compileAndImportPkg(t, "issue15920")
}

func TestIssue20046(t *testing.T) {
	// This package only handles gc export data.
	needsCompiler(t, "gc")

	// On windows, we have to set the -D option for the compiler to avoid having a drive
	// letter and an illegal ':' in the import path - just skip it (see also issue #3483).
	if runtime.GOOS == "windows" {
		t.Skip("avoid dealing with relative paths/drive letters on windows")
	}

	// "./issue20046".V.M must exist
	pkg := compileAndImportPkg(t, "issue20046")
	obj := lookupObj(t, pkg.Scope(), "V")
	if m, index, indirect := types.LookupFieldOrMethod(obj.Type(), false, nil, "M"); m == nil {
		t.Fatalf("V.M not found (index = %v, indirect = %v)", index, indirect)
	}
}

func TestIssue25301(t *testing.T) {
	testenv.NeedsGo1Point(t, 11)
	// This package only handles gc export data.
	needsCompiler(t, "gc")

	// On windows, we have to set the -D option for the compiler to avoid having a drive
	// letter and an illegal ':' in the import path - just skip it (see also issue #3483).
	if runtime.GOOS == "windows" {
		t.Skip("avoid dealing with relative paths/drive letters on windows")
	}

	compileAndImportPkg(t, "issue25301")
}

func TestIssue51836(t *testing.T) {
	testenv.NeedsGo1Point(t, 18) // requires generics

	// This package only handles gc export data.
	needsCompiler(t, "gc")

	// On windows, we have to set the -D option for the compiler to avoid having a drive
	// letter and an illegal ':' in the import path - just skip it (see also issue #3483).
	if runtime.GOOS == "windows" {
		t.Skip("avoid dealing with relative paths/drive letters on windows")
	}

	tmpdir := mktmpdir(t)
	defer os.RemoveAll(tmpdir)
	testoutdir := filepath.Join(tmpdir, "testdata")

	dir := filepath.Join("testdata", "issue51836")
	// Following the pattern of TestIssue13898, aa.go needs to be compiled from
	// the output directory. We pass the full path to compile() so that we don't
	// have to copy the file to that directory.
	bpath, err := filepath.Abs(filepath.Join(dir, "aa.go"))
	if err != nil {
		t.Fatal(err)
	}
	compile(t, dir, "a.go", testoutdir)
	compile(t, testoutdir, bpath, testoutdir)

	// import must succeed (test for issue at hand)
	_ = importPkg(t, "./testdata/aa", tmpdir)
}

func importPkg(t *testing.T, path, srcDir string) *types.Package {
	pkg, err := Import(make(map[string]*types.Package), path, srcDir, nil)
	if err != nil {
		t.Fatal(err)
	}
	return pkg
}

func compileAndImportPkg(t *testing.T, name string) *types.Package {
	tmpdir := mktmpdir(t)
	defer os.RemoveAll(tmpdir)
	compile(t, "testdata", name+".go", filepath.Join(tmpdir, "testdata"))
	return importPkg(t, "./testdata/"+name, tmpdir)
}

func lookupObj(t *testing.T, scope *types.Scope, name string) types.Object {
	if obj := scope.Lookup(name); obj != nil {
		return obj
	}
	t.Fatalf("%s not found", name)
	return nil
}
