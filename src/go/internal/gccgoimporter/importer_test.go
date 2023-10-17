// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gccgoimporter

import (
	"go/types"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"testing"
)

type importerTest struct {
	pkgpath, name, want, wantval string
	wantinits                    []string
	gccgoVersion                 int // minimum gccgo version (0 => any)
}

func runImporterTest(t *testing.T, imp Importer, initmap map[*types.Package]InitData, test *importerTest) {
	pkg, err := imp(make(map[string]*types.Package), test.pkgpath, ".", nil)
	if err != nil {
		t.Error(err)
		return
	}

	if test.name != "" {
		obj := pkg.Scope().Lookup(test.name)
		if obj == nil {
			t.Errorf("%s: object not found", test.name)
			return
		}

		got := types.ObjectString(obj, types.RelativeTo(pkg))
		if got != test.want {
			t.Errorf("%s: got %q; want %q", test.name, got, test.want)
		}

		if test.wantval != "" {
			gotval := obj.(*types.Const).Val().String()
			if gotval != test.wantval {
				t.Errorf("%s: got val %q; want val %q", test.name, gotval, test.wantval)
			}
		}
	}

	if len(test.wantinits) > 0 {
		initdata := initmap[pkg]
		found := false
		// Check that the package's own init function has the package's priority
		for _, pkginit := range initdata.Inits {
			if pkginit.InitFunc == test.wantinits[0] {
				found = true
				break
			}
		}

		if !found {
			t.Errorf("%s: could not find expected function %q", test.pkgpath, test.wantinits[0])
		}

		// FIXME: the original version of this test was written against
		// the v1 export data scheme for capturing init functions, so it
		// verified the priority values. We moved away from the priority
		// scheme some time ago; it is not clear how much work it would be
		// to validate the new init export data.
	}
}

// When adding tests to this list, be sure to set the 'gccgoVersion'
// field if the testcases uses a "recent" Go addition (ex: aliases).
var importerTests = [...]importerTest{
	{pkgpath: "pointer", name: "Int8Ptr", want: "type Int8Ptr *int8"},
	{pkgpath: "complexnums", name: "NN", want: "const NN untyped complex", wantval: "(-1 + -1i)"},
	{pkgpath: "complexnums", name: "NP", want: "const NP untyped complex", wantval: "(-1 + 1i)"},
	{pkgpath: "complexnums", name: "PN", want: "const PN untyped complex", wantval: "(1 + -1i)"},
	{pkgpath: "complexnums", name: "PP", want: "const PP untyped complex", wantval: "(1 + 1i)"},
	{pkgpath: "conversions", name: "Bits", want: "const Bits Units", wantval: `"bits"`},
	{pkgpath: "time", name: "Duration", want: "type Duration int64"},
	{pkgpath: "time", name: "Nanosecond", want: "const Nanosecond Duration", wantval: "1"},
	{pkgpath: "unicode", name: "IsUpper", want: "func IsUpper(r rune) bool"},
	{pkgpath: "unicode", name: "MaxRune", want: "const MaxRune untyped rune", wantval: "1114111"},
	{pkgpath: "imports", wantinits: []string{"imports..import", "fmt..import"}},
	{pkgpath: "importsar", name: "Hello", want: "var Hello string"},
	{pkgpath: "aliases", name: "A14", gccgoVersion: 7, want: "type A14 = func(int, T0) chan T2"},
	{pkgpath: "aliases", name: "C0", gccgoVersion: 7, want: "type C0 struct{f1 C1; f2 C1}"},
	{pkgpath: "escapeinfo", name: "NewT", want: "func NewT(data []byte) *T"},
	{pkgpath: "issue27856", name: "M", gccgoVersion: 7, want: "type M struct{E F}"},
	{pkgpath: "v1reflect", name: "Type", want: "type Type interface{Align() int; AssignableTo(u Type) bool; Bits() int; ChanDir() ChanDir; Elem() Type; Field(i int) StructField; FieldAlign() int; FieldByIndex(index []int) StructField; FieldByName(name string) (StructField, bool); FieldByNameFunc(match func(string) bool) (StructField, bool); Implements(u Type) bool; In(i int) Type; IsVariadic() bool; Key() Type; Kind() Kind; Len() int; Method(int) Method; MethodByName(string) (Method, bool); Name() string; NumField() int; NumIn() int; NumMethod() int; NumOut() int; Out(i int) Type; PkgPath() string; Size() uintptr; String() string; common() *commonType; rawString() string; runtimeType() *runtimeType; uncommon() *uncommonType}"},
	{pkgpath: "nointerface", name: "I", want: "type I int"},
	{pkgpath: "issue29198", name: "FooServer", gccgoVersion: 7, want: "type FooServer struct{FooServer *FooServer; user string; ctx context.Context}"},
	{pkgpath: "issue30628", name: "Apple", want: "type Apple struct{hey sync.RWMutex; x int; RQ [517]struct{Count uintptr; NumBytes uintptr; Last uintptr}}"},
	{pkgpath: "issue31540", name: "S", gccgoVersion: 7, want: "type S struct{b int; map[Y]Z}"}, // should want "type S struct{b int; A2}" (issue  #44410)
	{pkgpath: "issue34182", name: "T1", want: "type T1 struct{f *T2}"},
	{pkgpath: "notinheap", name: "S", want: "type S struct{}"},
}

func TestGoxImporter(t *testing.T) {
	testenv.MustHaveExec(t)
	initmap := make(map[*types.Package]InitData)
	imp := GetImporter([]string{"testdata"}, initmap)

	for _, test := range importerTests {
		runImporterTest(t, imp, initmap, &test)
	}
}

// gccgoPath returns a path to gccgo if it is present (either in
// path or specified via GCCGO environment variable), or an
// empty string if no gccgo is available.
func gccgoPath() string {
	gccgoname := os.Getenv("GCCGO")
	if gccgoname == "" {
		gccgoname = "gccgo"
	}
	if gpath, gerr := exec.LookPath(gccgoname); gerr == nil {
		return gpath
	}
	return ""
}

func TestObjImporter(t *testing.T) {
	// This test relies on gccgo being around.
	gpath := gccgoPath()
	if gpath == "" {
		t.Skip("This test needs gccgo")
	}

	verout, err := testenv.Command(t, gpath, "--version").CombinedOutput()
	if err != nil {
		t.Logf("%s", verout)
		t.Fatal(err)
	}
	vers := regexp.MustCompile(`(\d+)\.(\d+)`).FindSubmatch(verout)
	if len(vers) == 0 {
		t.Fatalf("could not find version number in %s", verout)
	}
	major, err := strconv.Atoi(string(vers[1]))
	if err != nil {
		t.Fatal(err)
	}
	minor, err := strconv.Atoi(string(vers[2]))
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("gccgo version %d.%d", major, minor)

	tmpdir := t.TempDir()
	initmap := make(map[*types.Package]InitData)
	imp := GetImporter([]string{tmpdir}, initmap)

	artmpdir := t.TempDir()
	arinitmap := make(map[*types.Package]InitData)
	arimp := GetImporter([]string{artmpdir}, arinitmap)

	for _, test := range importerTests {
		if major < test.gccgoVersion {
			// Support for type aliases was added in GCC 7.
			t.Logf("skipping %q: not supported before gccgo version %d", test.pkgpath, test.gccgoVersion)
			continue
		}

		gofile := filepath.Join("testdata", test.pkgpath+".go")
		if _, err := os.Stat(gofile); os.IsNotExist(err) {
			continue
		}
		ofile := filepath.Join(tmpdir, test.pkgpath+".o")
		afile := filepath.Join(artmpdir, "lib"+test.pkgpath+".a")

		cmd := testenv.Command(t, gpath, "-fgo-pkgpath="+test.pkgpath, "-c", "-o", ofile, gofile)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Logf("%s", out)
			t.Fatalf("gccgo %s failed: %s", gofile, err)
		}

		runImporterTest(t, imp, initmap, &test)

		cmd = testenv.Command(t, "ar", "cr", afile, ofile)
		out, err = cmd.CombinedOutput()
		if err != nil {
			t.Logf("%s", out)
			t.Fatalf("ar cr %s %s failed: %s", afile, ofile, err)
		}

		runImporterTest(t, arimp, arinitmap, &test)

		if err = os.Remove(ofile); err != nil {
			t.Fatal(err)
		}
		if err = os.Remove(afile); err != nil {
			t.Fatal(err)
		}
	}
}
