// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is a copy of $GOROOT/src/go/internal/gcimporter/gcimporter_test.go,
// adjusted to make it build with code from (std lib) internal/testenv copied.

package gcimporter

import (
	"bytes"
	"fmt"
	"go/types"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

// ----------------------------------------------------------------------------
// The following three functions (Builder, HasGoBuild, MustHaveGoBuild) were
// copied from $GOROOT/src/internal/testenv since that package is not available
// in x/tools.

// Builder reports the name of the builder running this test
// (for example, "linux-amd64" or "windows-386-gce").
// If the test is not running on the build infrastructure,
// Builder returns the empty string.
func Builder() string {
	return os.Getenv("GO_BUILDER_NAME")
}

// HasGoBuild reports whether the current system can build programs with ``go build''
// and then run them with os.StartProcess or exec.Command.
func HasGoBuild() bool {
	switch runtime.GOOS {
	case "android", "nacl":
		return false
	case "darwin":
		if strings.HasPrefix(runtime.GOARCH, "arm") {
			return false
		}
	}
	return true
}

// MustHaveGoBuild checks that the current system can build programs with ``go build''
// and then run them with os.StartProcess or exec.Command.
// If not, MustHaveGoBuild calls t.Skip with an explanation.
func MustHaveGoBuild(t *testing.T) {
	if !HasGoBuild() {
		t.Skipf("skipping test: 'go build' not available on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
}

// ----------------------------------------------------------------------------

// skipSpecialPlatforms causes the test to be skipped for platforms where
// builders (build.golang.org) don't have access to compiled packages for
// import.
func skipSpecialPlatforms(t *testing.T) {
	switch platform := runtime.GOOS + "-" + runtime.GOARCH; platform {
	case "nacl-amd64p32",
		"nacl-386",
		"nacl-arm",
		"darwin-arm",
		"darwin-arm64":
		t.Skipf("no compiled packages available for import on %s", platform)
	}
}

func compile(t *testing.T, dirname, filename string) string {
	/* testenv. */ MustHaveGoBuild(t)
	cmd := exec.Command("go", "tool", "compile", filename)
	cmd.Dir = dirname
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Logf("%s", out)
		t.Fatalf("go tool compile %s failed: %s", filename, err)
	}
	// filename should end with ".go"
	return filepath.Join(dirname, filename[:len(filename)-2]+"o")
}

func testPath(t *testing.T, path, srcDir string) *types.Package {
	t0 := time.Now()
	pkg, err := Import(make(map[string]*types.Package), path, srcDir)
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

const testfile = "exports.go"

func TestImportTestdata(t *testing.T) {
	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	if outFn := compile(t, "testdata", testfile); outFn != "" {
		defer os.Remove(outFn)
	}

	// filename should end with ".go"
	filename := testfile[:len(testfile)-3]
	if pkg := testPath(t, "./testdata/"+filename, "."); pkg != nil {
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
	skipSpecialPlatforms(t) // we really only need to exclude nacl platforms, but this is fine

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	const dir = "./testdata/versions"
	list, err := ioutil.ReadDir(dir)
	if err != nil {
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

		// test that export data can be imported
		_, err := Import(make(map[string]*types.Package), pkgpath, dir)
		if err != nil {
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
		filename := filepath.Join(dir, pkgpath) + ".a"
		ioutil.WriteFile(filename, data, 0666)
		defer os.Remove(filename)

		// test that importing the corrupted file results in an error
		_, err = Import(make(map[string]*types.Package), pkgpath, dir)
		if err == nil {
			t.Errorf("import corrupted %q succeeded", pkgpath)
		} else if msg := err.Error(); !strings.Contains(msg, "version skew") {
			t.Errorf("import %q error incorrect (%s)", pkgpath, msg)
		}
	}
}

func TestImportStdLib(t *testing.T) {
	skipSpecialPlatforms(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	dt := maxTime
	if testing.Short() && /* testenv. */ Builder() == "" {
		dt = 10 * time.Millisecond
	}
	nimports := testDir(t, "", time.Now().Add(dt)) // installed packages
	t.Logf("tested %d imports", nimports)
}

var importedObjectTests = []struct {
	name string
	want string
}{
	{"math.Pi", "const Pi untyped float"},
	{"io.Reader", "type Reader interface{Read(p []byte) (n int, err error)}"},
	// Go 1.7 and 1.8 don't know about embedded interfaces. Leave this
	// test out for now - the code is tested in the std library anyway.
	// TODO(gri) enable again once we're off 1.7 and 1.8.
	// {"io.ReadWriter", "type ReadWriter interface{Reader; Writer}"},
	{"math.Sin", "func Sin(x float64) float64"},
	// TODO(gri) add more tests
}

func TestImportedTypes(t *testing.T) {
	skipSpecialPlatforms(t)

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

		pkg, err := Import(make(map[string]*types.Package), importPath, ".")
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
	}
}

func TestIssue5815(t *testing.T) {
	skipSpecialPlatforms(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	pkg := importPkg(t, "strings")

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
	skipSpecialPlatforms(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	imports := make(map[string]*types.Package)
	_, err := Import(imports, "net/http", ".")
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
	skipSpecialPlatforms(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	// On windows, we have to set the -D option for the compiler to avoid having a drive
	// letter and an illegal ':' in the import path - just skip it (see also issue #3483).
	if runtime.GOOS == "windows" {
		t.Skip("avoid dealing with relative paths/drive letters on windows")
	}

	if f := compile(t, "testdata", "a.go"); f != "" {
		defer os.Remove(f)
	}
	if f := compile(t, "testdata", "b.go"); f != "" {
		defer os.Remove(f)
	}

	// import must succeed (test for issue at hand)
	pkg := importPkg(t, "./testdata/b")

	// make sure all indirectly imported packages have names
	for _, imp := range pkg.Imports() {
		if imp.Name() == "" {
			t.Errorf("no name for %s package", imp.Path())
		}
	}
}

func TestIssue13898(t *testing.T) {
	skipSpecialPlatforms(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	// import go/internal/gcimporter which imports go/types partially
	imports := make(map[string]*types.Package)
	_, err := Import(imports, "go/internal/gcimporter", ".")
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
	skipSpecialPlatforms(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	// On windows, we have to set the -D option for the compiler to avoid having a drive
	// letter and an illegal ':' in the import path - just skip it (see also issue #3483).
	if runtime.GOOS == "windows" {
		t.Skip("avoid dealing with relative paths/drive letters on windows")
	}

	if f := compile(t, "testdata", "p.go"); f != "" {
		defer os.Remove(f)
	}

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
		if _, err := Import(imports, "./././testdata/p", "."); err != nil {
			t.Fatal(err)
		}
	}
}

func TestIssue15920(t *testing.T) {
	skipSpecialPlatforms(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	// On windows, we have to set the -D option for the compiler to avoid having a drive
	// letter and an illegal ':' in the import path - just skip it (see also issue #3483).
	if runtime.GOOS == "windows" {
		t.Skip("avoid dealing with relative paths/drive letters on windows")
	}

	if f := compile(t, "testdata", "issue15920.go"); f != "" {
		defer os.Remove(f)
	}

	importPkg(t, "./testdata/issue15920")
}

func TestIssue20046(t *testing.T) {
	skipSpecialPlatforms(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	// On windows, we have to set the -D option for the compiler to avoid having a drive
	// letter and an illegal ':' in the import path - just skip it (see also issue #3483).
	if runtime.GOOS == "windows" {
		t.Skip("avoid dealing with relative paths/drive letters on windows")
	}

	if f := compile(t, "testdata", "issue20046.go"); f != "" {
		defer os.Remove(f)
	}

	// "./issue20046".V.M must exist
	pkg := importPkg(t, "./testdata/issue20046")
	obj := lookupObj(t, pkg.Scope(), "V")
	if m, index, indirect := types.LookupFieldOrMethod(obj.Type(), false, nil, "M"); m == nil {
		t.Fatalf("V.M not found (index = %v, indirect = %v)", index, indirect)
	}
}

func importPkg(t *testing.T, path string) *types.Package {
	pkg, err := Import(make(map[string]*types.Package), path, ".")
	if err != nil {
		t.Fatal(err)
	}
	return pkg
}

func lookupObj(t *testing.T, scope *types.Scope, name string) types.Object {
	if obj := scope.Lookup(name); obj != nil {
		return obj
	}
	t.Fatalf("%s not found", name)
	return nil
}
