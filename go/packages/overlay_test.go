package packages_test

import (
	"fmt"
	"log"
	"path/filepath"
	"reflect"
	"testing"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/packages/packagestest"
)

const commonMode = packages.NeedName | packages.NeedFiles |
	packages.NeedCompiledGoFiles | packages.NeedImports | packages.NeedSyntax

func TestOverlayChangesPackage(t *testing.T) {
	log.SetFlags(log.Lshortfile)
	exported := packagestest.Export(t, packagestest.GOPATH, []packagestest.Module{{
		Name: "fake",
		Files: map[string]interface{}{
			"a.go": "package foo\nfunc f(){}\n",
		},
		Overlay: map[string][]byte{
			"a.go": []byte("package foox\nfunc f(){}\n"),
		},
	}})
	defer exported.Cleanup()
	exported.Config.Mode = packages.NeedName

	initial, err := packages.Load(exported.Config,
		filepath.Dir(exported.File("fake", "a.go")))
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}
	if len(initial) != 1 || initial[0].ID != "fake" || initial[0].Name != "foox" {
		t.Fatalf("got %v, expected [fake]", initial)
	}
	if len(initial[0].Errors) != 0 {
		t.Fatalf("got %v, expected no errors", initial[0].Errors)
	}
	log.SetFlags(0)
}
func TestOverlayChangesBothPackages(t *testing.T) {
	log.SetFlags(log.Lshortfile)
	exported := packagestest.Export(t, packagestest.GOPATH, []packagestest.Module{{
		Name: "fake",
		Files: map[string]interface{}{
			"a.go":      "package foo\nfunc g(){}\n",
			"a_test.go": "package foo\nfunc f(){}\n",
		},
		Overlay: map[string][]byte{
			"a.go":      []byte("package foox\nfunc g(){}\n"),
			"a_test.go": []byte("package foox\nfunc f(){}\n"),
		},
	}})
	defer exported.Cleanup()
	exported.Config.Mode = commonMode

	initial, err := packages.Load(exported.Config,
		filepath.Dir(exported.File("fake", "a.go")))
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}
	if len(initial) != 3 {
		t.Errorf("got %d packges, expected 3", len(initial))
	}
	want := []struct {
		id, name string
		count    int
	}{
		{"fake", "foox", 1},
		{"fake [fake.test]", "foox", 2},
		{"fake.test", "main", 1},
	}
	for i := 0; i < 3; i++ {
		if ok := checkPkg(t, initial[i], want[i].id, want[i].name, want[i].count); !ok {
			t.Errorf("%d: got {%s %s %d}, expected %v", i, initial[i].ID,
				initial[i].Name, len(initial[i].Syntax), want[i])
		}
		if len(initial[i].Errors) != 0 {
			t.Errorf("%d: got %v, expected no errors", i, initial[i].Errors)
		}
	}
	log.SetFlags(0)
}

func TestOverlayChangesTestPackage(t *testing.T) {
	log.SetFlags(log.Lshortfile)
	exported := packagestest.Export(t, packagestest.GOPATH, []packagestest.Module{{
		Name: "fake",
		Files: map[string]interface{}{
			"a_test.go": "package foo\nfunc f(){}\n",
		},
		Overlay: map[string][]byte{
			"a_test.go": []byte("package foox\nfunc f(){}\n"),
		},
	}})
	defer exported.Cleanup()
	exported.Config.Mode = commonMode

	initial, err := packages.Load(exported.Config,
		filepath.Dir(exported.File("fake", "a_test.go")))
	if err != nil {
		t.Fatalf("failed to load: %v", err)
	}
	if len(initial) != 3 {
		t.Errorf("got %d packges, expected 3", len(initial))
	}
	want := []struct {
		id, name string
		count    int
	}{
		{"fake", "foo", 0},
		{"fake [fake.test]", "foox", 1},
		{"fake.test", "main", 1},
	}
	for i := 0; i < 3; i++ {
		if ok := checkPkg(t, initial[i], want[i].id, want[i].name, want[i].count); !ok {
			t.Errorf("got {%s %s %d}, expected %v", initial[i].ID,
				initial[i].Name, len(initial[i].Syntax), want[i])
		}
	}
	if len(initial[0].Errors) != 0 {
		t.Fatalf("got %v, expected no errors", initial[0].Errors)
	}
	log.SetFlags(0)
}

func TestOverlayXTests(t *testing.T) {
	packagestest.TestAll(t, testOverlayXTests)
}

// This test checks the behavior of go/packages.Load with an overlaid
// x test. The source of truth is the go/packages.Load results for the
// exact same package, just on-disk.
func testOverlayXTests(t *testing.T, exporter packagestest.Exporter) {
	const aFile = `package a; const C = "C"; func Hello() {}`
	const aTestVariant = `package a

import "testing"

const TestC = "test" + C

func TestHello(){
	Hello()
}`
	const aXTest = `package a_test

import (
	"testing"

	"golang.org/fake/a"
)

const xTestC = "x" + a.C

func TestHello(t *testing.T) {
	a.Hello()
}`

	// First, get the source of truth by loading the package, all on disk.
	onDisk := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go":        aFile,
			"a/a_test.go":   aTestVariant,
			"a/a_x_test.go": aXTest,
		},
	}})
	defer onDisk.Cleanup()

	onDisk.Config.Mode = commonMode
	onDisk.Config.Tests = true
	onDisk.Config.Mode = packages.LoadTypes
	initial, err := packages.Load(onDisk.Config, fmt.Sprintf("file=%s", onDisk.File("golang.org/fake", "a/a_x_test.go")))
	if err != nil {
		t.Fatal(err)
	}
	wantPkg := initial[0]

	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go":        aFile,
			"a/a_test.go":   aTestVariant,
			"a/a_x_test.go": ``, // empty x test on disk
		},
		Overlay: map[string][]byte{
			"a/a_x_test.go": []byte(aXTest),
		},
	}})
	defer exported.Cleanup()

	if len(initial) != 1 {
		t.Fatalf("expected 1 package, got %d", len(initial))
	}
	// Confirm that the overlaid package is identical to the on-disk version.
	pkg := initial[0]
	if !reflect.DeepEqual(wantPkg, pkg) {
		t.Fatalf("mismatched packages: want %#v, got %#v", wantPkg, pkg)
	}
	xTestC := constant(pkg, "xTestC")
	if xTestC == nil {
		t.Fatalf("no value for xTestC")
	}
	got := xTestC.Val().String()
	// TODO(rstambler): Ideally, this test would check that the test variant
	// was imported, but that's pretty complicated.
	if want := `"xC"`; got != want {
		t.Errorf("got: %q, want %q", got, want)
	}
}

func checkPkg(t *testing.T, p *packages.Package, id, name string, syntax int) bool {
	t.Helper()
	if p.ID == id && p.Name == name && len(p.Syntax) == syntax {
		return true
	}
	return false
}
