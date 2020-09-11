package packages_test

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/packages/packagestest"
	"golang.org/x/tools/internal/testenv"
)

const (
	commonMode = packages.NeedName | packages.NeedFiles |
		packages.NeedCompiledGoFiles | packages.NeedImports | packages.NeedSyntax
	everythingMode = commonMode | packages.NeedDeps | packages.NeedTypes |
		packages.NeedTypesSizes
)

func TestOverlayChangesPackageName(t *testing.T) {
	packagestest.TestAll(t, testOverlayChangesPackageName)
}
func testOverlayChangesPackageName(t *testing.T, exporter packagestest.Exporter) {
	log.SetFlags(log.Lshortfile)
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
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
func TestOverlayChangesBothPackageNames(t *testing.T) {
	packagestest.TestAll(t, testOverlayChangesBothPackageNames)
}
func testOverlayChangesBothPackageNames(t *testing.T, exporter packagestest.Exporter) {
	log.SetFlags(log.Lshortfile)
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
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
func TestOverlayChangesTestPackageName(t *testing.T) {
	packagestest.TestAll(t, testOverlayChangesTestPackageName)
}
func testOverlayChangesTestPackageName(t *testing.T, exporter packagestest.Exporter) {
	log.SetFlags(log.Lshortfile)
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
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

func checkPkg(t *testing.T, p *packages.Package, id, name string, syntax int) bool {
	t.Helper()
	if p.ID == id && p.Name == name && len(p.Syntax) == syntax {
		return true
	}
	return false
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

func TestOverlay(t *testing.T) { packagestest.TestAll(t, testOverlay) }
func testOverlay(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go":      `package a; import "golang.org/fake/b"; const A = "a" + b.B`,
			"b/b.go":      `package b; import "golang.org/fake/c"; const B = "b" + c.C`,
			"c/c.go":      `package c; const C = "c"`,
			"c/c_test.go": `package c; import "testing"; func TestC(t *testing.T) {}`,
			"d/d.go":      `package d; const D = "d"`,
		}}})
	defer exported.Cleanup()

	for i, test := range []struct {
		overlay  map[string][]byte
		want     string // expected value of a.A
		wantErrs []string
	}{
		{nil, `"abc"`, nil},                 // default
		{map[string][]byte{}, `"abc"`, nil}, // empty overlay
		{map[string][]byte{exported.File("golang.org/fake", "c/c.go"): []byte(`package c; const C = "C"`)}, `"abC"`, nil},
		{map[string][]byte{exported.File("golang.org/fake", "b/b.go"): []byte(`package b; import "golang.org/fake/c"; const B = "B" + c.C`)}, `"aBc"`, nil},
		// Overlay with an existing file in an existing package adding a new import.
		{map[string][]byte{exported.File("golang.org/fake", "b/b.go"): []byte(`package b; import "golang.org/fake/d"; const B = "B" + d.D`)}, `"aBd"`, nil},
		// Overlay with an existing file in an existing package.
		{map[string][]byte{exported.File("golang.org/fake", "c/c.go"): []byte(`package c; import "net/http"; const C = http.MethodGet`)}, `"abGET"`, nil},
		// Overlay with a new file in an existing package.
		{map[string][]byte{
			exported.File("golang.org/fake", "c/c.go"):                                               []byte(`package c;`),
			filepath.Join(filepath.Dir(exported.File("golang.org/fake", "c/c.go")), "c_new_file.go"): []byte(`package c; const C = "Ç"`)},
			`"abÇ"`, nil},
		// Overlay with a new file in an existing package, adding a new dependency to that package.
		{map[string][]byte{
			exported.File("golang.org/fake", "c/c.go"):                                               []byte(`package c;`),
			filepath.Join(filepath.Dir(exported.File("golang.org/fake", "c/c.go")), "c_new_file.go"): []byte(`package c; import "golang.org/fake/d"; const C = "c" + d.D`)},
			`"abcd"`, nil},
	} {
		exported.Config.Overlay = test.overlay
		exported.Config.Mode = packages.LoadAllSyntax
		initial, err := packages.Load(exported.Config, "golang.org/fake/a")
		if err != nil {
			t.Error(err)
			continue
		}

		// Check value of a.A.
		a := initial[0]
		aA := constant(a, "A")
		if aA == nil {
			t.Errorf("%d. a.A: got nil", i)
			continue
		}
		got := aA.Val().String()
		if got != test.want {
			t.Errorf("%d. a.A: got %s, want %s", i, got, test.want)
		}

		// Check errors.
		var errors []packages.Error
		packages.Visit(initial, nil, func(pkg *packages.Package) {
			errors = append(errors, pkg.Errors...)
		})
		if errs := errorMessages(errors); !reflect.DeepEqual(errs, test.wantErrs) {
			t.Errorf("%d. got errors %s, want %s", i, errs, test.wantErrs)
		}
	}
}

func TestOverlayDeps(t *testing.T) { packagestest.TestAll(t, testOverlayDeps) }
func testOverlayDeps(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"c/c.go":      `package c; const C = "c"`,
			"c/c_test.go": `package c; import "testing"; func TestC(t *testing.T) {}`,
		},
	}})
	defer exported.Cleanup()

	exported.Config.Overlay = map[string][]byte{exported.File("golang.org/fake", "c/c.go"): []byte(`package c; import "net/http"; const C = http.MethodGet`)}
	exported.Config.Mode = packages.NeedName |
		packages.NeedFiles |
		packages.NeedCompiledGoFiles |
		packages.NeedImports |
		packages.NeedDeps |
		packages.NeedTypesSizes
	pkgs, err := packages.Load(exported.Config, fmt.Sprintf("file=%s", exported.File("golang.org/fake", "c/c.go")))
	if err != nil {
		t.Error(err)
	}

	// Find package golang.org/fake/c
	sort.Slice(pkgs, func(i, j int) bool { return pkgs[i].ID < pkgs[j].ID })
	pkgc := pkgs[0]
	if pkgc.ID != "golang.org/fake/c" {
		t.Errorf("expected first package in sorted list to be \"golang.org/fake/c\", got %v", pkgc.ID)
	}

	// Make sure golang.org/fake/c imports net/http, as per the overlay.
	contains := func(imports map[string]*packages.Package, wantImport string) bool {
		for imp := range imports {
			if imp == wantImport {
				return true
			}
		}
		return false
	}
	if !contains(pkgc.Imports, "net/http") {
		t.Errorf("expected import of %s in package %s, got the following imports: %v",
			"net/http", pkgc.ID, pkgc.Imports)
	}

}

func TestNewPackagesInOverlay(t *testing.T) { packagestest.TestAll(t, testNewPackagesInOverlay) }
func testNewPackagesInOverlay(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{
		{
			Name: "golang.org/fake",
			Files: map[string]interface{}{
				"a/a.go": `package a; import "golang.org/fake/b"; const A = "a" + b.B`,
				"b/b.go": `package b; import "golang.org/fake/c"; const B = "b" + c.C`,
				"c/c.go": `package c; const C = "c"`,
				"d/d.go": `package d; const D = "d"`,
			},
		},
		{
			Name: "example.com/extramodule",
			Files: map[string]interface{}{
				"pkg/x.go": "package pkg\n",
			},
		},
	})
	defer exported.Cleanup()

	dir := filepath.Dir(filepath.Dir(exported.File("golang.org/fake", "a/a.go")))

	for _, test := range []struct {
		name    string
		overlay map[string][]byte
		want    string // expected value of e.E
	}{
		{"one_file",
			map[string][]byte{
				filepath.Join(dir, "e", "e.go"): []byte(`package e; import "golang.org/fake/a"; const E = "e" + a.A`)},
			`"eabc"`},
		{"multiple_files_same_package",
			map[string][]byte{
				filepath.Join(dir, "e", "e.go"):      []byte(`package e; import "golang.org/fake/a"; const E = "e" + a.A + underscore`),
				filepath.Join(dir, "e", "e_util.go"): []byte(`package e; const underscore = "_"`),
			},
			`"eabc_"`},
		{"multiple_files_two_packages",
			map[string][]byte{
				filepath.Join(dir, "e", "e.go"):      []byte(`package e; import "golang.org/fake/f"; const E = "e" + f.F + underscore`),
				filepath.Join(dir, "e", "e_util.go"): []byte(`package e; const underscore = "_"`),
				filepath.Join(dir, "f", "f.go"):      []byte(`package f; const F = "f"`),
			},
			`"ef_"`},
		{"multiple_files_three_packages",
			map[string][]byte{
				filepath.Join(dir, "e", "e.go"):      []byte(`package e; import "golang.org/fake/f"; const E = "e" + f.F + underscore`),
				filepath.Join(dir, "e", "e_util.go"): []byte(`package e; const underscore = "_"`),
				filepath.Join(dir, "f", "f.go"):      []byte(`package f; import "golang.org/fake/g"; const F = "f" + g.G`),
				filepath.Join(dir, "g", "g.go"):      []byte(`package g; const G = "g"`),
			},
			`"efg_"`},
		{"multiple_files_four_packages",
			map[string][]byte{
				filepath.Join(dir, "e", "e.go"):      []byte(`package e; import "golang.org/fake/f"; import "golang.org/fake/h"; const E = "e" + f.F + h.H + underscore`),
				filepath.Join(dir, "e", "e_util.go"): []byte(`package e; const underscore = "_"`),
				filepath.Join(dir, "f", "f.go"):      []byte(`package f; import "golang.org/fake/g"; const F = "f" + g.G`),
				filepath.Join(dir, "g", "g.go"):      []byte(`package g; const G = "g"`),
				filepath.Join(dir, "h", "h.go"):      []byte(`package h; const H = "h"`),
			},
			`"efgh_"`},
		{"multiple_files_four_packages_again",
			map[string][]byte{
				filepath.Join(dir, "e", "e.go"):      []byte(`package e; import "golang.org/fake/f"; const E = "e" + f.F + underscore`),
				filepath.Join(dir, "e", "e_util.go"): []byte(`package e; const underscore = "_"`),
				filepath.Join(dir, "f", "f.go"):      []byte(`package f; import "golang.org/fake/g"; const F = "f" + g.G`),
				filepath.Join(dir, "g", "g.go"):      []byte(`package g; import "golang.org/fake/h"; const G = "g" + h.H`),
				filepath.Join(dir, "h", "h.go"):      []byte(`package h; const H = "h"`),
			},
			`"efgh_"`},
		{"main_overlay",
			map[string][]byte{
				filepath.Join(dir, "e", "main.go"): []byte(`package main; import "golang.org/fake/a"; const E = "e" + a.A; func main(){}`)},
			`"eabc"`},
	} {
		t.Run(test.name, func(t *testing.T) {
			exported.Config.Overlay = test.overlay
			exported.Config.Mode = packages.LoadAllSyntax
			exported.Config.Logf = t.Logf

			// With an overlay, we don't know the expected import path,
			// so load with the absolute path of the directory.
			initial, err := packages.Load(exported.Config, filepath.Join(dir, "e"))
			if err != nil {
				t.Fatal(err)
			}

			// Check value of e.E.
			e := initial[0]
			eE := constant(e, "E")
			if eE == nil {
				t.Fatalf("e.E: was nil in %#v", e)
			}
			got := eE.Val().String()
			if got != test.want {
				t.Fatalf("e.E: got %s, want %s", got, test.want)
			}
		})
	}
}

// Test that we can create a package and its test package in an overlay.
func TestOverlayNewPackageAndTest(t *testing.T) {
	packagestest.TestAll(t, testOverlayNewPackageAndTest)
}
func testOverlayNewPackageAndTest(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{
		{
			Name: "golang.org/fake",
			Files: map[string]interface{}{
				"foo.txt": "placeholder",
			},
		},
	})
	defer exported.Cleanup()

	dir := filepath.Dir(exported.File("golang.org/fake", "foo.txt"))
	exported.Config.Overlay = map[string][]byte{
		filepath.Join(dir, "a.go"):      []byte(`package a;`),
		filepath.Join(dir, "a_test.go"): []byte(`package a; import "testing";`),
	}
	initial, err := packages.Load(exported.Config, "file="+filepath.Join(dir, "a.go"), "file="+filepath.Join(dir, "a_test.go"))
	if err != nil {
		t.Fatal(err)
	}
	if len(initial) != 2 {
		t.Errorf("got %v packages, wanted %v", len(initial), 2)
	}
}

func TestAdHocOverlays(t *testing.T) {
	testenv.NeedsTool(t, "go")

	// This test doesn't use packagestest because we are testing ad-hoc packages,
	// which are outside of $GOPATH and outside of a module.
	tmp, err := ioutil.TempDir("", "testAdHocOverlays")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)

	filename := filepath.Join(tmp, "a.go")
	content := []byte(`package a
const A = 1
`)

	// Make sure that the user's value of GO111MODULE does not affect test results.
	for _, go111module := range []string{"off", "auto", "on"} {
		t.Run("GO111MODULE="+go111module, func(t *testing.T) {
			config := &packages.Config{
				Dir:  tmp,
				Env:  append(os.Environ(), "GOPACKAGESDRIVER=off", fmt.Sprintf("GO111MODULE=%s", go111module)),
				Mode: packages.LoadAllSyntax,
				Overlay: map[string][]byte{
					filename: content,
				},
				Logf: t.Logf,
			}
			initial, err := packages.Load(config, fmt.Sprintf("file=%s", filename))
			if err != nil {
				t.Fatal(err)
			}
			if len(initial) == 0 {
				t.Fatalf("no packages for %s", filename)
			}
			// Check value of a.A.
			a := initial[0]
			if a.Errors != nil {
				t.Fatalf("a: got errors %+v, want no error", err)
			}
			aA := constant(a, "A")
			if aA == nil {
				t.Errorf("a.A: got nil")
				return
			}
			got := aA.Val().String()
			if want := "1"; got != want {
				t.Errorf("a.A: got %s, want %s", got, want)
			}
		})
	}
}

// TestOverlayModFileChanges tests the behavior resulting from having files
// from multiple modules in overlays.
func TestOverlayModFileChanges(t *testing.T) {
	testenv.NeedsTool(t, "go")

	// Create two unrelated modules in a temporary directory.
	tmp, err := ioutil.TempDir("", "tmp")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)

	// mod1 has a dependency on golang.org/x/xerrors.
	mod1, err := ioutil.TempDir(tmp, "mod1")
	if err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(mod1, "go.mod"), []byte(`module mod1

	require (
		golang.org/x/xerrors v0.0.0-20190717185122-a985d3407aa7
	)
	`), 0775); err != nil {
		t.Fatal(err)
	}

	// mod2 does not have any dependencies.
	mod2, err := ioutil.TempDir(tmp, "mod2")
	if err != nil {
		t.Fatal(err)
	}

	want := `module mod2

go 1.11
`
	if err := ioutil.WriteFile(filepath.Join(mod2, "go.mod"), []byte(want), 0775); err != nil {
		t.Fatal(err)
	}

	// Run packages.Load on mod2, while passing the contents over mod1/main.go in the overlay.
	config := &packages.Config{
		Dir:  mod2,
		Env:  append(os.Environ(), "GOPACKAGESDRIVER=off"),
		Mode: packages.LoadImports,
		Overlay: map[string][]byte{
			filepath.Join(mod1, "main.go"): []byte(`package main
import "golang.org/x/xerrors"
func main() {
	_ = errors.New("")
}
`),
			filepath.Join(mod2, "main.go"): []byte(`package main
func main() {}
`),
		},
	}
	if _, err := packages.Load(config, fmt.Sprintf("file=%s", filepath.Join(mod2, "main.go"))); err != nil {
		t.Fatal(err)
	}

	// Check that mod2/go.mod has not been modified.
	got, err := ioutil.ReadFile(filepath.Join(mod2, "go.mod"))
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != want {
		t.Errorf("expected %s, got %s", want, string(got))
	}
}

func TestOverlayGOPATHVendoring(t *testing.T) {
	exported := packagestest.Export(t, packagestest.GOPATH, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"vendor/vendor.com/foo/foo.go": `package foo; const X = "hi"`,
			"user/user.go":                 `package user`,
		},
	}})
	defer exported.Cleanup()

	exported.Config.Mode = packages.LoadAllSyntax
	exported.Config.Logf = t.Logf
	exported.Config.Overlay = map[string][]byte{
		exported.File("golang.org/fake", "user/user.go"): []byte(`package user; import "vendor.com/foo"; var x = foo.X`),
	}
	initial, err := packages.Load(exported.Config, "golang.org/fake/user")
	if err != nil {
		t.Fatal(err)
	}
	user := initial[0]
	if len(user.Imports) != 1 {
		t.Fatal("no imports for user")
	}
	if user.Imports["vendor.com/foo"].Name != "foo" {
		t.Errorf("failed to load vendored package foo, imports: %#v", user.Imports["vendor.com/foo"])
	}
}

func TestContainsOverlay(t *testing.T) { packagestest.TestAll(t, testContainsOverlay) }
func testContainsOverlay(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go": `package a; import "golang.org/fake/b"`,
			"b/b.go": `package b; import "golang.org/fake/c"`,
			"c/c.go": `package c`,
		}}})
	defer exported.Cleanup()
	bOverlayFile := filepath.Join(filepath.Dir(exported.File("golang.org/fake", "b/b.go")), "b_overlay.go")
	exported.Config.Mode = packages.LoadImports
	exported.Config.Overlay = map[string][]byte{bOverlayFile: []byte(`package b;`)}
	initial, err := packages.Load(exported.Config, "file="+bOverlayFile)
	if err != nil {
		t.Fatal(err)
	}

	graph, _ := importGraph(initial)
	wantGraph := `
* golang.org/fake/b
  golang.org/fake/c
  golang.org/fake/b -> golang.org/fake/c
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}
}

func TestContainsOverlayXTest(t *testing.T) { packagestest.TestAll(t, testContainsOverlayXTest) }
func testContainsOverlayXTest(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{{
		Name: "golang.org/fake",
		Files: map[string]interface{}{
			"a/a.go": `package a; import "golang.org/fake/b"`,
			"b/b.go": `package b; import "golang.org/fake/c"`,
			"c/c.go": `package c`,
		}}})
	defer exported.Cleanup()

	bOverlayXTestFile := filepath.Join(filepath.Dir(exported.File("golang.org/fake", "b/b.go")), "b_overlay_x_test.go")
	exported.Config.Mode = packages.NeedName | packages.NeedFiles | packages.NeedImports
	exported.Config.Overlay = map[string][]byte{bOverlayXTestFile: []byte(`package b_test; import "golang.org/fake/b"`)}
	initial, err := packages.Load(exported.Config, "file="+bOverlayXTestFile)
	if err != nil {
		t.Fatal(err)
	}

	graph, _ := importGraph(initial)
	wantGraph := `
  golang.org/fake/b
* golang.org/fake/b_test [golang.org/fake/b.test]
  golang.org/fake/c
  golang.org/fake/b -> golang.org/fake/c
  golang.org/fake/b_test [golang.org/fake/b.test] -> golang.org/fake/b
`[1:]
	if graph != wantGraph {
		t.Errorf("wrong import graph: got <<%s>>, want <<%s>>", graph, wantGraph)
	}
}

func TestInvalidFilesBeforeOverlay(t *testing.T) {
	packagestest.TestAll(t, testInvalidFilesBeforeOverlay)
}

func testInvalidFilesBeforeOverlay(t *testing.T, exporter packagestest.Exporter) {
	testenv.NeedsGo1Point(t, 15)

	exported := packagestest.Export(t, exporter, []packagestest.Module{
		{
			Name: "golang.org/fake",
			Files: map[string]interface{}{
				"d/d.go":  ``,
				"main.go": ``,
			},
		},
	})
	defer exported.Cleanup()

	dir := filepath.Dir(filepath.Dir(exported.File("golang.org/fake", "d/d.go")))

	exported.Config.Mode = everythingMode
	exported.Config.Tests = true

	// First, check that all packages returned have files associated with them.
	// Tests the work-around for golang/go#39986.
	t.Run("no overlay", func(t *testing.T) {
		initial, err := packages.Load(exported.Config, fmt.Sprintf("%s/...", dir))
		if err != nil {
			t.Fatal(err)
		}
		for _, pkg := range initial {
			if len(pkg.CompiledGoFiles) == 0 {
				t.Fatalf("expected at least 1 CompiledGoFile for %s, got none", pkg.PkgPath)
			}
		}
	})

}

// Tests golang/go#35973, fixed in Go 1.14.
func TestInvalidFilesBeforeOverlayContains(t *testing.T) {
	packagestest.TestAll(t, testInvalidFilesBeforeOverlayContains)
}
func testInvalidFilesBeforeOverlayContains(t *testing.T, exporter packagestest.Exporter) {
	testenv.NeedsGo1Point(t, 15)

	exported := packagestest.Export(t, exporter, []packagestest.Module{
		{
			Name: "golang.org/fake",
			Files: map[string]interface{}{
				"d/d.go":      `package d; import "net/http"; const Get = http.MethodGet; const Hello = "hello";`,
				"d/util.go":   ``,
				"d/d_test.go": ``,
				"main.go":     ``,
			},
		},
	})
	defer exported.Cleanup()

	dir := filepath.Dir(filepath.Dir(exported.File("golang.org/fake", "d/d.go")))

	// Additional tests for test variants.
	for i, tt := range []struct {
		name    string
		overlay map[string][]byte
		want    string // expected value of d.D
		wantID  string // expected value for the package ID
	}{
		// Overlay with a test variant.
		{
			"test_variant",
			map[string][]byte{
				filepath.Join(dir, "d", "d_test.go"): []byte(`package d; import "testing"; const D = Get + "_test"; func TestD(t *testing.T) {};`),
			},
			`"GET_test"`, "golang.org/fake/d [golang.org/fake/d.test]",
		},
		// Overlay in package.
		{
			"second_file",
			map[string][]byte{
				filepath.Join(dir, "d", "util.go"): []byte(`package d; const D = Get + "_util";`),
			},
			`"GET_util"`, "golang.org/fake/d",
		},
		// Overlay on the main file.
		{
			"main",
			map[string][]byte{
				filepath.Join(dir, "main.go"): []byte(`package main; import "golang.org/fake/d"; const D = d.Get + "_main"; func main() {};`),
			},
			`"GET_main"`, "golang.org/fake",
		},
		{
			"xtest",
			map[string][]byte{
				filepath.Join(dir, "d", "d_test.go"): []byte(`package d_test; import "golang.org/fake/d"; import "testing"; const D = d.Get + "_xtest"; func TestD(t *testing.T) {};`),
			},
			`"GET_xtest"`, "golang.org/fake/d_test [golang.org/fake/d.test]",
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			exported.Config.Overlay = tt.overlay
			exported.Config.Mode = everythingMode
			exported.Config.Tests = true

			for f := range tt.overlay {
				initial, err := packages.Load(exported.Config, fmt.Sprintf("file=%s", f))
				if err != nil {
					t.Fatal(err)
				}
				pkg := initial[0]
				if pkg.ID != tt.wantID {
					t.Fatalf("expected package ID %q, got %q", tt.wantID, pkg.ID)
				}
				var containsFile bool
				for _, goFile := range pkg.CompiledGoFiles {
					if f == goFile {
						containsFile = true
						break
					}
				}
				if !containsFile {
					t.Fatalf("expected %s in CompiledGoFiles, got %v", f, pkg.CompiledGoFiles)
				}
				// Check value of d.D.
				D := constant(pkg, "D")
				if D == nil {
					t.Fatalf("%d. D: got nil", i)
				}
				got := D.Val().String()
				if got != tt.want {
					t.Fatalf("%d. D: got %s, want %s", i, got, tt.want)
				}
			}
		})
	}
}

func TestInvalidXTestInGOPATH(t *testing.T) {
	packagestest.TestAll(t, testInvalidXTestInGOPATH)
}
func testInvalidXTestInGOPATH(t *testing.T, exporter packagestest.Exporter) {
	t.Skip("Not fixed yet. See golang.org/issue/40825.")

	exported := packagestest.Export(t, exporter, []packagestest.Module{
		{
			Name: "golang.org/fake",
			Files: map[string]interface{}{
				"x/x.go":      `package x`,
				"x/x_test.go": ``,
			},
		},
	})
	defer exported.Cleanup()

	dir := filepath.Dir(filepath.Dir(exported.File("golang.org/fake", "x/x.go")))

	exported.Config.Mode = everythingMode
	exported.Config.Tests = true

	initial, err := packages.Load(exported.Config, fmt.Sprintf("%s/...", dir))
	if err != nil {
		t.Fatal(err)
	}
	pkg := initial[0]
	if len(pkg.CompiledGoFiles) != 2 {
		t.Fatalf("expected at least 2 CompiledGoFiles for %s, got %v", pkg.PkgPath, len(pkg.CompiledGoFiles))
	}
}

// Reproduces golang/go#40685.
func TestAddImportInOverlay(t *testing.T) {
	packagestest.TestAll(t, testAddImportInOverlay)
}
func testAddImportInOverlay(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{
		{
			Name: "golang.org/fake",
			Files: map[string]interface{}{
				"a/a.go": `package a

import (
	"fmt"
)

func _() {
	fmt.Println("")
	os.Stat("")
}`,
				"a/a_test.go": `package a

import (
	"os"
	"testing"
)

func TestA(t *testing.T) {
	os.Stat("")
}`,
			},
		},
	})
	defer exported.Cleanup()

	exported.Config.Mode = everythingMode
	exported.Config.Tests = true

	dir := filepath.Dir(exported.File("golang.org/fake", "a/a.go"))
	exported.Config.Overlay = map[string][]byte{
		filepath.Join(dir, "a.go"): []byte(`package a

import (
	"fmt"
	"os"
)

func _() {
	fmt.Println("")
	os.Stat("")
}
`),
	}
	initial, err := packages.Load(exported.Config, "golang.org/fake/a")
	if err != nil {
		t.Fatal(err)
	}
	pkg := initial[0]
	var foundOs bool
	for _, imp := range pkg.Imports {
		if imp.PkgPath == "os" {
			foundOs = true
			break
		}
	}
	if !foundOs {
		t.Fatalf(`expected import "os", found none: %v`, pkg.Imports)
	}
}

// Tests that overlays are applied for different kinds of load patterns.
func TestLoadDifferentPatterns(t *testing.T) {
	packagestest.TestAll(t, testLoadDifferentPatterns)
}
func testLoadDifferentPatterns(t *testing.T, exporter packagestest.Exporter) {
	exported := packagestest.Export(t, exporter, []packagestest.Module{
		{
			Name: "golang.org/fake",
			Files: map[string]interface{}{
				"foo.txt": "placeholder",
				"b/b.go": `package b
import "golang.org/fake/a"
func _() {
	a.Hi()
}
`,
			},
		},
	})
	defer exported.Cleanup()

	exported.Config.Mode = everythingMode
	exported.Config.Tests = true

	dir := filepath.Dir(exported.File("golang.org/fake", "foo.txt"))
	exported.Config.Overlay = map[string][]byte{
		filepath.Join(dir, "a", "a.go"): []byte(`package a
import "fmt"
func Hi() {
	fmt.Println("")
}
`),
	}
	for _, tc := range []struct {
		pattern string
	}{
		{"golang.org/fake/a"},
		{"golang.org/fake/..."},
		{fmt.Sprintf("file=%s", filepath.Join(dir, "a", "a.go"))},
	} {
		t.Run(tc.pattern, func(t *testing.T) {
			initial, err := packages.Load(exported.Config, tc.pattern)
			if err != nil {
				t.Fatal(err)
			}
			var match *packages.Package
			for _, pkg := range initial {
				if pkg.PkgPath == "golang.org/fake/a" {
					match = pkg
					break
				}
			}
			if match == nil {
				t.Fatalf(`expected package path "golang.org/fake/a", got %q`, match.PkgPath)
			}
			if match.PkgPath != "golang.org/fake/a" {
				t.Fatalf(`expected package path "golang.org/fake/a", got %q`, match.PkgPath)
			}
			if _, ok := match.Imports["fmt"]; !ok {
				t.Fatalf(`expected import "fmt", got none`)
			}
		})
	}

	// Now, load "golang.org/fake/b" and confirm that "golang.org/fake/a" is
	// not returned as a root.
	initial, err := packages.Load(exported.Config, "golang.org/fake/b")
	if err != nil {
		t.Fatal(err)
	}
	if len(initial) > 1 {
		t.Fatalf("expected 1 package, got %v", initial)
	}
	pkg := initial[0]
	if pkg.PkgPath != "golang.org/fake/b" {
		t.Fatalf(`expected package path "golang.org/fake/b", got %q`, pkg.PkgPath)
	}
	if _, ok := pkg.Imports["golang.org/fake/a"]; !ok {
		t.Fatalf(`expected import "golang.org/fake/a", got none`)
	}
}

// Tests that overlays are applied for a replaced module.
// This does not use go/packagestest because it needs to write a replace
// directive with an absolute path in one of the module's go.mod files.
func TestOverlaysInReplace(t *testing.T) {
	// Create module b.com in a temporary directory. Do not add any Go files
	// on disk.
	tmpPkgs, err := ioutil.TempDir("", "modules")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpPkgs)

	dirB := filepath.Join(tmpPkgs, "b")
	if err := os.Mkdir(dirB, 0775); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(dirB, "go.mod"), []byte(fmt.Sprintf("module %s.com", dirB)), 0775); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(dirB, "inner"), 0775); err != nil {
		t.Fatal(err)
	}

	// Create a separate module that requires and replaces b.com.
	tmpWorkspace, err := ioutil.TempDir("", "workspace")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpWorkspace)
	goModContent := fmt.Sprintf(`module workspace.com

require (
	b.com v0.0.0-00010101000000-000000000000
)

replace (
	b.com => %s
)
`, dirB)
	if err := ioutil.WriteFile(filepath.Join(tmpWorkspace, "go.mod"), []byte(goModContent), 0775); err != nil {
		t.Fatal(err)
	}

	// Add Go files for b.com/inner in an overlay and try loading it from the
	// workspace.com module.
	config := &packages.Config{
		Dir:  tmpWorkspace,
		Mode: packages.LoadAllSyntax,
		Logf: t.Logf,
		Overlay: map[string][]byte{
			filepath.Join(dirB, "inner", "b.go"): []byte(`package inner; import "fmt"; func _() { fmt.Println("");`),
		},
	}
	initial, err := packages.Load(config, "b.com/...")
	if err != nil {
		t.Error(err)
	}
	pkg := initial[0]
	if pkg.PkgPath != "b.com/inner" {
		t.Fatalf(`expected package path "b.com/inner", got %q`, pkg.PkgPath)
	}
	if _, ok := pkg.Imports["fmt"]; !ok {
		t.Fatalf(`expected import "fmt", got none`)
	}
}
