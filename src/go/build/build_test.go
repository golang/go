// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"flag"
	"internal/testenv"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

func TestMain(m *testing.M) {
	flag.Parse()
	if goTool, err := testenv.GoTool(); err == nil {
		os.Setenv("PATH", filepath.Dir(goTool)+string(os.PathListSeparator)+os.Getenv("PATH"))
	}
	os.Exit(m.Run())
}

func TestMatch(t *testing.T) {
	ctxt := Default
	what := "default"
	match := func(tag string, want map[string]bool) {
		m := make(map[string]bool)
		if !ctxt.match(tag, m) {
			t.Errorf("%s context should match %s, does not", what, tag)
		}
		if !reflect.DeepEqual(m, want) {
			t.Errorf("%s tags = %v, want %v", tag, m, want)
		}
	}
	nomatch := func(tag string, want map[string]bool) {
		m := make(map[string]bool)
		if ctxt.match(tag, m) {
			t.Errorf("%s context should NOT match %s, does", what, tag)
		}
		if !reflect.DeepEqual(m, want) {
			t.Errorf("%s tags = %v, want %v", tag, m, want)
		}
	}

	match(runtime.GOOS+","+runtime.GOARCH, map[string]bool{runtime.GOOS: true, runtime.GOARCH: true})
	match(runtime.GOOS+","+runtime.GOARCH+",!foo", map[string]bool{runtime.GOOS: true, runtime.GOARCH: true, "foo": true})
	nomatch(runtime.GOOS+","+runtime.GOARCH+",foo", map[string]bool{runtime.GOOS: true, runtime.GOARCH: true, "foo": true})

	what = "modified"
	ctxt.BuildTags = []string{"foo"}
	match(runtime.GOOS+","+runtime.GOARCH, map[string]bool{runtime.GOOS: true, runtime.GOARCH: true})
	match(runtime.GOOS+","+runtime.GOARCH+",foo", map[string]bool{runtime.GOOS: true, runtime.GOARCH: true, "foo": true})
	nomatch(runtime.GOOS+","+runtime.GOARCH+",!foo", map[string]bool{runtime.GOOS: true, runtime.GOARCH: true, "foo": true})
	match(runtime.GOOS+","+runtime.GOARCH+",!bar", map[string]bool{runtime.GOOS: true, runtime.GOARCH: true, "bar": true})
	nomatch(runtime.GOOS+","+runtime.GOARCH+",bar", map[string]bool{runtime.GOOS: true, runtime.GOARCH: true, "bar": true})
	nomatch("!", map[string]bool{})
}

func TestDotSlashImport(t *testing.T) {
	p, err := ImportDir("testdata/other", 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(p.Imports) != 1 || p.Imports[0] != "./file" {
		t.Fatalf("testdata/other: Imports=%v, want [./file]", p.Imports)
	}

	p1, err := Import("./file", "testdata/other", 0)
	if err != nil {
		t.Fatal(err)
	}
	if p1.Name != "file" {
		t.Fatalf("./file: Name=%q, want %q", p1.Name, "file")
	}
	dir := filepath.Clean("testdata/other/file") // Clean to use \ on Windows
	if p1.Dir != dir {
		t.Fatalf("./file: Dir=%q, want %q", p1.Name, dir)
	}
}

func TestEmptyImport(t *testing.T) {
	p, err := Import("", Default.GOROOT, FindOnly)
	if err == nil {
		t.Fatal(`Import("") returned nil error.`)
	}
	if p == nil {
		t.Fatal(`Import("") returned nil package.`)
	}
	if p.ImportPath != "" {
		t.Fatalf("ImportPath=%q, want %q.", p.ImportPath, "")
	}
}

func TestEmptyFolderImport(t *testing.T) {
	_, err := Import(".", "testdata/empty", 0)
	if _, ok := err.(*NoGoError); !ok {
		t.Fatal(`Import("testdata/empty") did not return NoGoError.`)
	}
}

func TestMultiplePackageImport(t *testing.T) {
	_, err := Import(".", "testdata/multi", 0)
	mpe, ok := err.(*MultiplePackageError)
	if !ok {
		t.Fatal(`Import("testdata/multi") did not return MultiplePackageError.`)
	}
	want := &MultiplePackageError{
		Dir:      filepath.FromSlash("testdata/multi"),
		Packages: []string{"main", "test_package"},
		Files:    []string{"file.go", "file_appengine.go"},
	}
	if !reflect.DeepEqual(mpe, want) {
		t.Errorf("got %#v; want %#v", mpe, want)
	}
}

func TestLocalDirectory(t *testing.T) {
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		t.Skipf("skipping on %s/%s, no valid GOROOT", runtime.GOOS, runtime.GOARCH)
	}

	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}

	p, err := ImportDir(cwd, 0)
	if err != nil {
		t.Fatal(err)
	}
	if p.ImportPath != "go/build" {
		t.Fatalf("ImportPath=%q, want %q", p.ImportPath, "go/build")
	}
}

func TestShouldBuild(t *testing.T) {
	const file1 = "// +build tag1\n\n" +
		"package main\n"
	want1 := map[string]bool{"tag1": true}

	const file2 = "// +build cgo\n\n" +
		"// This package implements parsing of tags like\n" +
		"// +build tag1\n" +
		"package build"
	want2 := map[string]bool{"cgo": true}

	const file3 = "// Copyright The Go Authors.\n\n" +
		"package build\n\n" +
		"// shouldBuild checks tags given by lines of the form\n" +
		"// +build tag\n" +
		"func shouldBuild(content []byte)\n"
	want3 := map[string]bool{}

	ctx := &Context{BuildTags: []string{"tag1"}}
	m := map[string]bool{}
	if !ctx.shouldBuild([]byte(file1), m, nil) {
		t.Errorf("shouldBuild(file1) = false, want true")
	}
	if !reflect.DeepEqual(m, want1) {
		t.Errorf("shouldBuild(file1) tags = %v, want %v", m, want1)
	}

	m = map[string]bool{}
	if ctx.shouldBuild([]byte(file2), m, nil) {
		t.Errorf("shouldBuild(file2) = true, want false")
	}
	if !reflect.DeepEqual(m, want2) {
		t.Errorf("shouldBuild(file2) tags = %v, want %v", m, want2)
	}

	m = map[string]bool{}
	ctx = &Context{BuildTags: nil}
	if !ctx.shouldBuild([]byte(file3), m, nil) {
		t.Errorf("shouldBuild(file3) = false, want true")
	}
	if !reflect.DeepEqual(m, want3) {
		t.Errorf("shouldBuild(file3) tags = %v, want %v", m, want3)
	}
}

func TestGoodOSArchFile(t *testing.T) {
	ctx := &Context{BuildTags: []string{"linux"}, GOOS: "darwin"}
	m := map[string]bool{}
	want := map[string]bool{"linux": true}
	if !ctx.goodOSArchFile("hello_linux.go", m) {
		t.Errorf("goodOSArchFile(hello_linux.go) = false, want true")
	}
	if !reflect.DeepEqual(m, want) {
		t.Errorf("goodOSArchFile(hello_linux.go) tags = %v, want %v", m, want)
	}
}

type readNopCloser struct {
	io.Reader
}

func (r readNopCloser) Close() error {
	return nil
}

var (
	ctxtP9      = Context{GOARCH: "arm", GOOS: "plan9"}
	ctxtAndroid = Context{GOARCH: "arm", GOOS: "android"}
)

var matchFileTests = []struct {
	ctxt  Context
	name  string
	data  string
	match bool
}{
	{ctxtP9, "foo_arm.go", "", true},
	{ctxtP9, "foo1_arm.go", "// +build linux\n\npackage main\n", false},
	{ctxtP9, "foo_darwin.go", "", false},
	{ctxtP9, "foo.go", "", true},
	{ctxtP9, "foo1.go", "// +build linux\n\npackage main\n", false},
	{ctxtP9, "foo.badsuffix", "", false},
	{ctxtAndroid, "foo_linux.go", "", true},
	{ctxtAndroid, "foo_android.go", "", true},
	{ctxtAndroid, "foo_plan9.go", "", false},
	{ctxtAndroid, "android.go", "", true},
	{ctxtAndroid, "plan9.go", "", true},
	{ctxtAndroid, "plan9_test.go", "", true},
	{ctxtAndroid, "arm.s", "", true},
	{ctxtAndroid, "amd64.s", "", true},
}

func TestMatchFile(t *testing.T) {
	for _, tt := range matchFileTests {
		ctxt := tt.ctxt
		ctxt.OpenFile = func(path string) (r io.ReadCloser, err error) {
			if path != "x+"+tt.name {
				t.Fatalf("OpenFile asked for %q, expected %q", path, "x+"+tt.name)
			}
			return &readNopCloser{strings.NewReader(tt.data)}, nil
		}
		ctxt.JoinPath = func(elem ...string) string {
			return strings.Join(elem, "+")
		}
		match, err := ctxt.MatchFile("x", tt.name)
		if match != tt.match || err != nil {
			t.Fatalf("MatchFile(%q) = %v, %v, want %v, nil", tt.name, match, err, tt.match)
		}
	}
}

func TestImportCmd(t *testing.T) {
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		t.Skipf("skipping on %s/%s, no valid GOROOT", runtime.GOOS, runtime.GOARCH)
	}

	p, err := Import("cmd/internal/objfile", "", 0)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(filepath.ToSlash(p.Dir), "src/cmd/internal/objfile") {
		t.Fatalf("Import cmd/internal/objfile returned Dir=%q, want %q", filepath.ToSlash(p.Dir), ".../src/cmd/internal/objfile")
	}
}

var (
	expandSrcDirPath = filepath.Join(string(filepath.Separator)+"projects", "src", "add")
)

var expandSrcDirTests = []struct {
	input, expected string
}{
	{"-L ${SRCDIR}/libs -ladd", "-L /projects/src/add/libs -ladd"},
	{"${SRCDIR}/add_linux_386.a -pthread -lstdc++", "/projects/src/add/add_linux_386.a -pthread -lstdc++"},
	{"Nothing to expand here!", "Nothing to expand here!"},
	{"$", "$"},
	{"$$", "$$"},
	{"${", "${"},
	{"$}", "$}"},
	{"$FOO ${BAR}", "$FOO ${BAR}"},
	{"Find me the $SRCDIRECTORY.", "Find me the $SRCDIRECTORY."},
	{"$SRCDIR is missing braces", "$SRCDIR is missing braces"},
}

func TestExpandSrcDir(t *testing.T) {
	for _, test := range expandSrcDirTests {
		output, _ := expandSrcDir(test.input, expandSrcDirPath)
		if output != test.expected {
			t.Errorf("%q expands to %q with SRCDIR=%q when %q is expected", test.input, output, expandSrcDirPath, test.expected)
		} else {
			t.Logf("%q expands to %q with SRCDIR=%q", test.input, output, expandSrcDirPath)
		}
	}
}

func TestShellSafety(t *testing.T) {
	tests := []struct {
		input, srcdir, expected string
		result                  bool
	}{
		{"-I${SRCDIR}/../include", "/projects/src/issue 11868", "-I/projects/src/issue 11868/../include", true},
		{"-I${SRCDIR}", "~wtf$@%^", "-I~wtf$@%^", true},
		{"-X${SRCDIR}/1,${SRCDIR}/2", "/projects/src/issue 11868", "-X/projects/src/issue 11868/1,/projects/src/issue 11868/2", true},
		{"-I/tmp -I/tmp", "/tmp2", "-I/tmp -I/tmp", true},
		{"-I/tmp", "/tmp/[0]", "-I/tmp", true},
		{"-I${SRCDIR}/dir", "/tmp/[0]", "-I/tmp/[0]/dir", false},
		{"-I${SRCDIR}/dir", "/tmp/go go", "-I/tmp/go go/dir", true},
		{"-I${SRCDIR}/dir dir", "/tmp/go", "-I/tmp/go/dir dir", true},
	}
	for _, test := range tests {
		output, ok := expandSrcDir(test.input, test.srcdir)
		if ok != test.result {
			t.Errorf("Expected %t while %q expands to %q with SRCDIR=%q; got %t", test.result, test.input, output, test.srcdir, ok)
		}
		if output != test.expected {
			t.Errorf("Expected %q while %q expands with SRCDIR=%q; got %q", test.expected, test.input, test.srcdir, output)
		}
	}
}

// Want to get a "cannot find package" error when directory for package does not exist.
// There should be valid partial information in the returned non-nil *Package.
func TestImportDirNotExist(t *testing.T) {
	testenv.MustHaveGoBuild(t) // really must just have source
	ctxt := Default

	emptyDir, err := ioutil.TempDir("", t.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(emptyDir)

	ctxt.GOPATH = emptyDir
	ctxt.Dir = emptyDir

	tests := []struct {
		label        string
		path, srcDir string
		mode         ImportMode
	}{
		{"Import(full, 0)", "go/build/doesnotexist", "", 0},
		{"Import(local, 0)", "./doesnotexist", filepath.Join(ctxt.GOROOT, "src/go/build"), 0},
		{"Import(full, FindOnly)", "go/build/doesnotexist", "", FindOnly},
		{"Import(local, FindOnly)", "./doesnotexist", filepath.Join(ctxt.GOROOT, "src/go/build"), FindOnly},
	}

	defer os.Setenv("GO111MODULE", os.Getenv("GO111MODULE"))

	for _, GO111MODULE := range []string{"off", "on"} {
		t.Run("GO111MODULE="+GO111MODULE, func(t *testing.T) {
			os.Setenv("GO111MODULE", GO111MODULE)

			for _, test := range tests {
				p, err := ctxt.Import(test.path, test.srcDir, test.mode)

				errOk := (err != nil && strings.HasPrefix(err.Error(), "cannot find package"))
				wantErr := `"cannot find package" error`
				if test.srcDir == "" {
					if err != nil && strings.Contains(err.Error(), "is not in GOROOT") {
						errOk = true
					}
					wantErr = `"cannot find package" or "is not in GOROOT" error`
				}
				if !errOk {
					t.Errorf("%s got error: %q, want %s", test.label, err, wantErr)
				}
				// If an error occurs, build.Import is documented to return
				// a non-nil *Package containing partial information.
				if p == nil {
					t.Fatalf(`%s got nil p, want non-nil *Package`, test.label)
				}
				// Verify partial information in p.
				if p.ImportPath != "go/build/doesnotexist" {
					t.Errorf(`%s got p.ImportPath: %q, want "go/build/doesnotexist"`, test.label, p.ImportPath)
				}
			}
		})
	}
}

func TestImportVendor(t *testing.T) {
	testenv.MustHaveGoBuild(t) // really must just have source

	defer os.Setenv("GO111MODULE", os.Getenv("GO111MODULE"))
	os.Setenv("GO111MODULE", "off")

	ctxt := Default
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	ctxt.GOPATH = filepath.Join(wd, "testdata/withvendor")
	p, err := ctxt.Import("c/d", filepath.Join(ctxt.GOPATH, "src/a/b"), 0)
	if err != nil {
		t.Fatalf("cannot find vendored c/d from testdata src/a/b directory: %v", err)
	}
	want := "a/vendor/c/d"
	if p.ImportPath != want {
		t.Fatalf("Import succeeded but found %q, want %q", p.ImportPath, want)
	}
}

func TestImportVendorFailure(t *testing.T) {
	testenv.MustHaveGoBuild(t) // really must just have source

	defer os.Setenv("GO111MODULE", os.Getenv("GO111MODULE"))
	os.Setenv("GO111MODULE", "off")

	ctxt := Default
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	ctxt.GOPATH = filepath.Join(wd, "testdata/withvendor")
	p, err := ctxt.Import("x.com/y/z", filepath.Join(ctxt.GOPATH, "src/a/b"), 0)
	if err == nil {
		t.Fatalf("found made-up package x.com/y/z in %s", p.Dir)
	}

	e := err.Error()
	if !strings.Contains(e, " (vendor tree)") {
		t.Fatalf("error on failed import does not mention GOROOT/src/vendor directory:\n%s", e)
	}
}

func TestImportVendorParentFailure(t *testing.T) {
	testenv.MustHaveGoBuild(t) // really must just have source

	defer os.Setenv("GO111MODULE", os.Getenv("GO111MODULE"))
	os.Setenv("GO111MODULE", "off")

	ctxt := Default
	wd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	ctxt.GOPATH = filepath.Join(wd, "testdata/withvendor")
	// This import should fail because the vendor/c directory has no source code.
	p, err := ctxt.Import("c", filepath.Join(ctxt.GOPATH, "src/a/b"), 0)
	if err == nil {
		t.Fatalf("found empty parent in %s", p.Dir)
	}
	if p != nil && p.Dir != "" {
		t.Fatalf("decided to use %s", p.Dir)
	}
	e := err.Error()
	if !strings.Contains(e, " (vendor tree)") {
		t.Fatalf("error on failed import does not mention GOROOT/src/vendor directory:\n%s", e)
	}
}

// Check that a package is loaded in module mode if GO111MODULE=on, even when
// no go.mod file is present. It should fail to resolve packages outside std.
// Verifies golang.org/issue/34669.
func TestImportPackageOutsideModule(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// Disable module fetching for this test so that 'go list' fails quickly
	// without trying to find the latest version of a module.
	defer os.Setenv("GOPROXY", os.Getenv("GOPROXY"))
	os.Setenv("GOPROXY", "off")

	// Create a GOPATH in a temporary directory. We don't use testdata
	// because it's in GOROOT, which interferes with the module heuristic.
	gopath, err := ioutil.TempDir("", "gobuild-notmodule")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(gopath)
	if err := os.MkdirAll(filepath.Join(gopath, "src/example.com/p"), 0777); err != nil {
		t.Fatal(err)
	}
	if err := ioutil.WriteFile(filepath.Join(gopath, "src/example.com/p/p.go"), []byte("package p"), 0666); err != nil {
		t.Fatal(err)
	}

	defer os.Setenv("GO111MODULE", os.Getenv("GO111MODULE"))
	os.Setenv("GO111MODULE", "on")
	defer os.Setenv("GOPATH", os.Getenv("GOPATH"))
	os.Setenv("GOPATH", gopath)
	ctxt := Default
	ctxt.GOPATH = gopath
	ctxt.Dir = filepath.Join(gopath, "src/example.com/p")

	want := "cannot find module providing package"
	if _, err := ctxt.Import("example.com/p", gopath, FindOnly); err == nil {
		t.Fatal("importing package when no go.mod is present succeeded unexpectedly")
	} else if errStr := err.Error(); !strings.Contains(errStr, want) {
		t.Fatalf("error when importing package when no go.mod is present: got %q; want %q", errStr, want)
	}
}

func TestImportDirTarget(t *testing.T) {
	testenv.MustHaveGoBuild(t) // really must just have source
	ctxt := Default
	ctxt.GOPATH = ""
	p, err := ctxt.ImportDir(filepath.Join(ctxt.GOROOT, "src/path"), 0)
	if err != nil {
		t.Fatal(err)
	}
	if p.PkgTargetRoot == "" || p.PkgObj == "" {
		t.Errorf("p.PkgTargetRoot == %q, p.PkgObj == %q, want non-empty", p.PkgTargetRoot, p.PkgObj)
	}
}

// TestIssue23594 prevents go/build from regressing and populating Package.Doc
// from comments in test files.
func TestIssue23594(t *testing.T) {
	// Package testdata/doc contains regular and external test files
	// with comments attached to their package declarations. The names of the files
	// ensure that we see the comments from the test files first.
	p, err := ImportDir("testdata/doc", 0)
	if err != nil {
		t.Fatalf("could not import testdata: %v", err)
	}

	if p.Doc != "Correct" {
		t.Fatalf("incorrectly set .Doc to %q", p.Doc)
	}
}

// TestMissingImportErrorRepetition checks that when an unknown package is
// imported, the package path is only shown once in the error.
// Verifies golang.org/issue/34752.
func TestMissingImportErrorRepetition(t *testing.T) {
	testenv.MustHaveGoBuild(t) // need 'go list' internally
	tmp, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)
	if err := ioutil.WriteFile(filepath.Join(tmp, "go.mod"), []byte("module m"), 0666); err != nil {
		t.Fatal(err)
	}
	defer os.Setenv("GO111MODULE", os.Getenv("GO111MODULE"))
	os.Setenv("GO111MODULE", "on")
	defer os.Setenv("GOPROXY", os.Getenv("GOPROXY"))
	os.Setenv("GOPROXY", "off")
	defer os.Setenv("GONOPROXY", os.Getenv("GONOPROXY"))
	os.Setenv("GONOPROXY", "none")

	ctxt := Default
	ctxt.Dir = tmp

	pkgPath := "example.com/hello"
	_, err = ctxt.Import(pkgPath, tmp, FindOnly)
	if err == nil {
		t.Fatal("unexpected success")
	}
	// Don't count the package path with a URL like https://...?go-get=1.
	// See golang.org/issue/35986.
	errStr := strings.ReplaceAll(err.Error(), "://"+pkgPath+"?go-get=1", "://...?go-get=1")
	if n := strings.Count(errStr, pkgPath); n != 1 {
		t.Fatalf("package path %q appears in error %d times; should appear once\nerror: %v", pkgPath, n, err)
	}
}

// TestCgoImportsIgnored checks that imports in cgo files are not included
// in the imports list when cgo is disabled.
// Verifies golang.org/issue/35946.
func TestCgoImportsIgnored(t *testing.T) {
	ctxt := Default
	ctxt.CgoEnabled = false
	p, err := ctxt.ImportDir("testdata/cgo_disabled", 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range p.Imports {
		if path == "should/be/ignored" {
			t.Errorf("found import %q in ignored cgo file", path)
		}
	}
}
