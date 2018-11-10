// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gccgoimporter

import (
	"go/types"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

type importerTest struct {
	pkgpath, name, want, wantval string
	wantinits                    []string
}

func runImporterTest(t *testing.T, imp Importer, initmap map[*types.Package]InitData, test *importerTest) {
	pkg, err := imp(make(map[string]*types.Package), test.pkgpath)
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
				if initdata.Priority != pkginit.Priority {
					t.Errorf("%s: got self priority %d; want %d", test.pkgpath, pkginit.Priority, initdata.Priority)
				}
				found = true
				break
			}
		}

		if !found {
			t.Errorf("%s: could not find expected function %q", test.pkgpath, test.wantinits[0])
		}

		// Each init function in the list other than the first one is a
		// dependency of the function immediately before it. Check that
		// the init functions appear in descending priority order.
		priority := initdata.Priority
		for _, wantdepinit := range test.wantinits[1:] {
			found = false
			for _, pkginit := range initdata.Inits {
				if pkginit.InitFunc == wantdepinit {
					if priority <= pkginit.Priority {
						t.Errorf("%s: got dep priority %d; want less than %d", test.pkgpath, pkginit.Priority, priority)
					}
					found = true
					priority = pkginit.Priority
					break
				}
			}

			if !found {
				t.Errorf("%s: could not find expected function %q", test.pkgpath, wantdepinit)
			}
		}
	}
}

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
	{pkgpath: "imports", wantinits: []string{"imports..import", "fmt..import", "math..import"}},
	{pkgpath: "alias", name: "IntAlias2", want: "type IntAlias2 = Int"},
}

func TestGoxImporter(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	initmap := make(map[*types.Package]InitData)
	imp := GetImporter([]string{"testdata"}, initmap)

	for _, test := range importerTests {
		runImporterTest(t, imp, initmap, &test)
	}
}

func TestObjImporter(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This test relies on gccgo being around, which it most likely will be if we
	// were compiled with gccgo.
	if runtime.Compiler != "gccgo" {
		t.Skip("This test needs gccgo")
		return
	}

	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatal(err)
	}
	initmap := make(map[*types.Package]InitData)
	imp := GetImporter([]string{tmpdir}, initmap)

	artmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatal(err)
	}
	arinitmap := make(map[*types.Package]InitData)
	arimp := GetImporter([]string{artmpdir}, arinitmap)

	for _, test := range importerTests {
		gofile := filepath.Join("testdata", test.pkgpath+".go")
		ofile := filepath.Join(tmpdir, test.pkgpath+".o")
		afile := filepath.Join(artmpdir, "lib"+test.pkgpath+".a")

		cmd := exec.Command("gccgo", "-fgo-pkgpath="+test.pkgpath, "-c", "-o", ofile, gofile)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Logf("%s", out)
			t.Fatalf("gccgo %s failed: %s", gofile, err)
		}

		runImporterTest(t, imp, initmap, &test)

		cmd = exec.Command("ar", "cr", afile, ofile)
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

	if err = os.Remove(tmpdir); err != nil {
		t.Fatal(err)
	}
}
