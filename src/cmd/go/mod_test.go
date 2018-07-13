// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"bytes"
	"internal/testenv"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	"cmd/go/internal/cfg"
	"cmd/go/internal/modconv"
	"cmd/go/internal/modload"
	"cmd/go/internal/txtar"
)

var cmdGoDir, _ = os.Getwd()

// testGoModules returns a testgoData set up for running
// tests of Go modules. It:
//
// - sets $GO111MODULE=on
// - sets $GOPROXY to the URL of a module proxy serving from ./testdata/mod
// - creates a new temp directory with subdirectories home, gopath, and work
// - sets $GOPATH to the new temp gopath directory
// - sets $HOME to the new temp home directory
// - writes work/go.mod containing "module m"
// - chdirs to the the new temp work directory
//
// The caller must defer tg.cleanup().
//
func testGoModules(t *testing.T) *testgoData {
	tg := testgo(t)
	tg.setenv("GO111MODULE", "on")
	StartProxy()
	tg.setenv("GOPROXY", proxyURL)
	tg.makeTempdir()
	tg.setenv(homeEnvName(), tg.path("home")) // for build cache
	tg.setenv("GOPATH", tg.path("gopath"))    // for download cache
	tg.tempFile("work/go.mod", "module m")
	tg.cd(tg.path("work"))

	return tg
}

// extract clears the temp work directory and then
// extracts the txtar archive named by file into that directory.
// The file name is interpreted relative to the cmd/go directory,
// so it usually begins with "testdata/".
func (tg *testgoData) extract(file string) {
	a, err := txtar.ParseFile(filepath.Join(cmdGoDir, file))
	if err != nil {
		tg.t.Fatal(err)
	}
	tg.cd(tg.path("."))
	tg.must(removeAll(tg.path("work")))
	tg.must(os.MkdirAll(tg.path("work"), 0777))
	tg.cd(tg.path("work"))
	for _, f := range a.Files {
		tg.tempFile(filepath.Join("work", f.Name), string(f.Data))
	}
}

func TestModGO111MODULE(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.tempFile("gp/src/x/y/z/go.mod", "module x/y/z")
	tg.tempFile("gp/src/x/y/z/w/w.txt", "")
	tg.tempFile("gp/foo/go.mod", "module example.com/mod")
	tg.tempFile("gp/foo/bar/baz/quux.txt", "")
	tg.tempFile("gp/bar/x.txt", "")
	tg.setenv("GOPATH", tg.path("gp"))

	// In GOPATH/src with go.mod.
	tg.cd(tg.path("gp/src/x/y/z"))
	tg.setenv("GO111MODULE", "auto")
	tg.run("env", "GOMOD")
	tg.grepStdoutNot(`go.mod`, "expected module mode disabled")

	tg.cd(tg.path("gp/src/x/y/z/w"))
	tg.run("env", "GOMOD")
	tg.grepStdoutNot(`go.mod`, "expected module mode disabled")

	tg.setenv("GO111MODULE", "off")
	tg.run("env", "GOMOD")
	tg.grepStdoutNot(`go.mod`, "expected module mode disabled")

	tg.setenv("GO111MODULE", "on")
	tg.run("env", "GOMOD")
	tg.grepStdout(`.*z[/\\]go.mod$`, "expected module mode enabled")

	// In GOPATH/src without go.mod.
	tg.cd(tg.path("gp/src/x/y"))
	tg.setenv("GO111MODULE", "auto")
	tg.run("env", "GOMOD")
	tg.grepStdoutNot(`go.mod`, "expected module mode disabled")

	tg.setenv("GO111MODULE", "off")
	tg.run("env", "GOMOD")
	tg.grepStdoutNot(`go.mod`, "expected module mode disabled")

	tg.setenv("GO111MODULE", "on")
	tg.runFail("env", "GOMOD")
	tg.grepStderr(`cannot find main module root`, "expected module mode failure")

	// Outside GOPATH/src with go.mod.
	tg.cd(tg.path("gp/foo"))
	tg.setenv("GO111MODULE", "auto")
	tg.run("env", "GOMOD")
	tg.grepStdout(`.*foo[/\\]go.mod$`, "expected module mode enabled")

	tg.cd(tg.path("gp/foo/bar/baz"))
	tg.run("env", "GOMOD")
	tg.grepStdout(`.*foo[/\\]go.mod$`, "expected module mode enabled")

	tg.setenv("GO111MODULE", "off")
	tg.run("env", "GOMOD")
	tg.grepStdoutNot(`go.mod`, "expected module mode disabled")
}

func TestModVersionsInGOPATHMode(t *testing.T) {
	tg := testgo(t)
	tg.setenv("GO111MODULE", "off") // GOPATH mode
	defer tg.cleanup()
	tg.makeTempdir()

	tg.runFail("get", "rsc.io/quote@v1.5.1")
	tg.grepStderr(`go: cannot use path@version syntax in GOPATH mode`, "expected path@version error")

	tg.runFail("build", "rsc.io/quote@v1.5.1")
	tg.grepStderr(`can't load package:.* cannot use path@version syntax in GOPATH mode`, "expected path@version error")

	tg.setenv("GO111MODULE", "on") // GOPATH mode
	tg.tempFile("x/go.mod", "module x")
	tg.cd(tg.path("x"))
	tg.runFail("build", "rsc.io/quote@v1.5.1")
	tg.grepStderr(`can't load package:.* can only use path@version syntax with 'go get'`, "expected path@version error")
}

func TestModFindModuleRoot(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.must(os.MkdirAll(tg.path("x/Godeps"), 0777))
	tg.must(os.MkdirAll(tg.path("x/vendor"), 0777))
	tg.must(os.MkdirAll(tg.path("x/y/z"), 0777))
	tg.must(os.MkdirAll(tg.path("x/.git"), 0777))
	var files []string
	for file := range modconv.Converters {
		files = append(files, file)
	}
	files = append(files, "go.mod")
	files = append(files, ".git/config")
	sort.Strings(files)

	for file := range modconv.Converters {
		tg.must(ioutil.WriteFile(tg.path("x/"+file), []byte{}, 0666))
		root, file1 := modload.FindModuleRoot(tg.path("x/y/z"), tg.path("."), true)
		if root != tg.path("x") || file1 != file {
			t.Errorf("%s: findModuleRoot = %q, %q, want %q, %q", file, root, file1, tg.path("x"), file)
		}
		tg.must(os.Remove(tg.path("x/" + file)))
	}
}

func TestModFindModulePath(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.must(os.MkdirAll(tg.path("x"), 0777))
	tg.must(ioutil.WriteFile(tg.path("x/x.go"), []byte("package x // import \"x\"\n"), 0666))
	path, err := modload.FindModulePath(tg.path("x"))
	if err != nil {
		t.Fatal(err)
	}
	if path != "x" {
		t.Fatalf("FindModulePath = %q, want %q", path, "x")
	}

	// Windows line-ending.
	tg.must(ioutil.WriteFile(tg.path("x/x.go"), []byte("package x // import \"x\"\r\n"), 0666))
	path, err = modload.FindModulePath(tg.path("x"))
	if path != "x" || err != nil {
		t.Fatalf("FindModulePath = %q, %v, want %q, nil", path, err, "x")
	}

	// Explicit setting in Godeps.json takes priority over implicit setting from GOPATH location.
	tg.tempFile("gp/src/example.com/x/y/z/z.go", "package z")
	gopath := cfg.BuildContext.GOPATH
	defer func() {
		cfg.BuildContext.GOPATH = gopath
	}()
	cfg.BuildContext.GOPATH = tg.path("gp")
	path, err = modload.FindModulePath(tg.path("gp/src/example.com/x/y/z"))
	if path != "example.com/x/y/z" || err != nil {
		t.Fatalf("FindModulePath = %q, %v, want %q, nil", path, err, "example.com/x/y/z")
	}

	tg.tempFile("gp/src/example.com/x/y/z/Godeps/Godeps.json", `
		{"ImportPath": "unexpected.com/z"}
	`)
	path, err = modload.FindModulePath(tg.path("gp/src/example.com/x/y/z"))
	if path != "unexpected.com/z" || err != nil {
		t.Fatalf("FindModulePath = %q, %v, want %q, nil", path, err, "unexpected.com/z")
	}

	// Empty dir outside GOPATH
	tg.must(os.MkdirAll(tg.path("gp1"), 0777))
	tg.must(os.MkdirAll(tg.path("x1"), 0777))
	cfg.BuildContext.GOPATH = tg.path("gp1")

	path, err = modload.FindModulePath(tg.path("x1"))
	if path != "" || err == nil {
		t.Fatalf("FindModulePath() = %q, %v, want %q, %q", path, err, "", "cannot determine module path for source directory")
	}

	// Empty dir inside GOPATH
	tg.must(os.MkdirAll(tg.path("gp2/src/x"), 0777))
	cfg.BuildContext.GOPATH = tg.path("gp2")

	path, err = modload.FindModulePath(tg.path("gp2/src/x"))
	if path != "x" || err != nil {
		t.Fatalf("FindModulePath() = %q, %v, want %q, nil", path, err, "x")
	}

	if !testenv.HasSymlink() {
		t.Logf("skipping symlink tests")
		return
	}

	// Empty dir inside GOPATH, dir has symlink
	// GOPATH = gp
	// gplink -> gp
	tg.must(os.MkdirAll(tg.path("gp3/src/x"), 0777))
	tg.must(os.Symlink(tg.path("gp3"), tg.path("gplink3")))
	cfg.BuildContext.GOPATH = tg.path("gp3")

	path, err = modload.FindModulePath(tg.path("gplink3/src/x"))
	if path != "x" || err != nil {
		t.Fatalf("FindModulePath() = %q, %v, want %q, nil", path, err, "x")
	}
	path, err = modload.FindModulePath(tg.path("gp3/src/x"))
	if path != "x" || err != nil {
		t.Fatalf("FindModulePath() = %q, %v, want %q, nil", path, err, "x")
	}

	// Empty dir inside GOPATH, dir has symlink 2
	// GOPATH = gp
	// gp/src/x -> x/x
	tg.must(os.MkdirAll(tg.path("gp4/src"), 0777))
	tg.must(os.MkdirAll(tg.path("x4/x"), 0777))
	tg.must(os.Symlink(tg.path("x4/x"), tg.path("gp4/src/x")))
	cfg.BuildContext.GOPATH = tg.path("gp4")

	path, err = modload.FindModulePath(tg.path("gp4/src/x"))
	if path != "x" || err != nil {
		t.Fatalf("FindModulePath() = %q, %v, want %q, nil", path, err, "x")
	}

	// Empty dir inside GOPATH, GOPATH has symlink
	// GOPATH = gplink
	// gplink -> gp
	tg.must(os.MkdirAll(tg.path("gp5/src/x"), 0777))
	tg.must(os.Symlink(tg.path("gp5"), tg.path("gplink5")))
	cfg.BuildContext.GOPATH = tg.path("gplink5")

	path, err = modload.FindModulePath(tg.path("gplink5/src/x"))
	if path != "x" || err != nil {
		t.Fatalf("FindModulePath() = %q, %v, want %q, nil", path, err, "x")
	}
	path, err = modload.FindModulePath(tg.path("gp5/src/x"))
	if path != "x" || err != nil {
		t.Fatalf("FindModulePath() = %q, %v, want %q, nil", path, err, "x")
	}

	// Empty dir inside GOPATH, GOPATH has symlink, dir has symlink 2
	// GOPATH = gplink
	// gplink -> gp
	// gplink2 -> gp
	tg.must(os.MkdirAll(tg.path("gp6/src/x"), 0777))
	tg.must(os.Symlink(tg.path("gp6"), tg.path("gplink6")))
	tg.must(os.Symlink(tg.path("gp6"), tg.path("gplink62")))
	cfg.BuildContext.GOPATH = tg.path("gplink6")

	path, err = modload.FindModulePath(tg.path("gplink62/src/x"))
	if path != "x" || err != nil {
		t.Fatalf("FindModulePath() = %q, %v, want %q, nil", path, err, "x")
	}
	path, err = modload.FindModulePath(tg.path("gplink6/src/x"))
	if path != "x" || err != nil {
		t.Fatalf("FindModulePath() = %q, %v, want %q, nil", path, err, "x")
	}
	path, err = modload.FindModulePath(tg.path("gp6/src/x"))
	if path != "x" || err != nil {
		t.Fatalf("FindModulePath() = %q, %v, want %q, nil", path, err, "x")
	}

	// Empty dir inside GOPATH, GOPATH has symlink, dir has symlink 3
	// GOPATH = gplink
	// gplink -> gp
	// gplink2 -> gp
	// gp/src/x -> x/x
	tg.must(os.MkdirAll(tg.path("gp7/src"), 0777))
	tg.must(os.MkdirAll(tg.path("x7/x"), 0777))
	tg.must(os.Symlink(tg.path("gp7"), tg.path("gplink7")))
	tg.must(os.Symlink(tg.path("gp7"), tg.path("gplink72")))
	tg.must(os.Symlink(tg.path("x7/x"), tg.path("gp7/src/x")))
	cfg.BuildContext.GOPATH = tg.path("gplink7")

	path, err = modload.FindModulePath(tg.path("gplink7/src/x"))
	if path != "x" || err != nil {
		t.Fatalf("FindModulePath() = %q, %v, want %q, nil", path, err, "x")
	}

	// This test fails when /tmp -> /private/tmp.
	// path, err = modload.FindModulePath(tg.path("gp7/src/x"))
	// if path != "x" || err != nil {
	// 	t.Fatalf("FindModulePath() = %q, %v, want %q, nil", path, err, "x")
	// }
}

func TestModImportModFails(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.setenv("GO111MODULE", "off")
	tg.tempFile("gopath/src/mod/foo/foo.go", "package foo")
	tg.runFail("list", "mod/foo")
	tg.grepStderr(`disallowed import path`, "expected disallowed because of module cache")
}

func TestModEdit(t *testing.T) {
	// Test that local replacements work
	// and that they can use a dummy name
	// that isn't resolvable and need not even
	// include a dot. See golang.org/issue/24100.
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.cd(tg.path("."))
	tg.must(os.MkdirAll(tg.path("w"), 0777))
	tg.must(ioutil.WriteFile(tg.path("x.go"), []byte("package x\n"), 0666))
	tg.must(ioutil.WriteFile(tg.path("w/w.go"), []byte("package w\n"), 0666))

	mustHaveGoMod := func(text string) {
		t.Helper()
		data, err := ioutil.ReadFile(tg.path("go.mod"))
		tg.must(err)
		if string(data) != text {
			t.Fatalf("go.mod mismatch:\nhave:<<<\n%s>>>\nwant:<<<\n%s\n", string(data), text)
		}
	}

	tg.runFail("mod", "-init")
	tg.grepStderr(`cannot determine module path`, "")
	_, err := os.Stat(tg.path("go.mod"))
	if err == nil {
		t.Fatalf("failed go mod -init created go.mod")
	}

	tg.run("mod", "-init", "-module", "x.x/y/z")
	tg.grepStderr("creating new go.mod: module x.x/y/z", "")
	mustHaveGoMod(`module x.x/y/z
`)

	tg.runFail("mod", "-init")
	mustHaveGoMod(`module x.x/y/z
`)

	tg.run("mod",
		"-droprequire=x.1",
		"-require=x.1@v1.0.0",
		"-require=x.2@v1.1.0",
		"-droprequire=x.2",
		"-exclude=x.1 @ v1.2.0",
		"-exclude=x.1@v1.2.1",
		"-replace=x.1@v1.3.0=y.1@v1.4.0",
		"-replace=x.1@v1.4.0 = ../z",
	)
	mustHaveGoMod(`module x.x/y/z

require x.1 v1.0.0

exclude (
	x.1 v1.2.0
	x.1 v1.2.1
)

replace (
	x.1 v1.3.0 => y.1 v1.4.0
	x.1 v1.4.0 => ../z
)
`)

	tg.run("mod",
		"-droprequire=x.1",
		"-dropexclude=x.1@v1.2.1",
		"-dropreplace=x.1@v1.3.0",
		"-require=x.3@v1.99.0",
	)
	mustHaveGoMod(`module x.x/y/z

exclude x.1 v1.2.0

replace x.1 v1.4.0 => ../z

require x.3 v1.99.0
`)

	tg.run("mod", "-json")
	want := `{
	"Module": {
		"Path": "x.x/y/z"
	},
	"Require": [
		{
			"Path": "x.3",
			"Version": "v1.99.0"
		}
	],
	"Exclude": [
		{
			"Path": "x.1",
			"Version": "v1.2.0"
		}
	],
	"Replace": [
		{
			"Old": {
				"Path": "x.1",
				"Version": "v1.4.0"
			},
			"New": {
				"Path": "../z"
			}
		}
	]
}
`
	if have := tg.getStdout(); have != want {
		t.Fatalf("go mod -json mismatch:\nhave:<<<\n%s>>>\nwant:<<<\n%s\n", have, want)
	}

	tg.run("mod",
		"-replace=x.1@v1.3.0=y.1/v2@v2.3.5",
		"-replace=x.1@v1.4.0=y.1/v2@v2.3.5",
	)
	mustHaveGoMod(`module x.x/y/z

exclude x.1 v1.2.0

replace (
	x.1 v1.3.0 => y.1/v2 v2.3.5
	x.1 v1.4.0 => y.1/v2 v2.3.5
)

require x.3 v1.99.0
`)
	tg.run("mod",
		"-replace=x.1=y.1/v2@v2.3.6",
	)
	mustHaveGoMod(`module x.x/y/z

exclude x.1 v1.2.0

replace x.1 => y.1/v2 v2.3.6

require x.3 v1.99.0
`)

	tg.run("mod", "-packages")
	want = `x.x/y/z
x.x/y/z/w
`
	if have := tg.getStdout(); have != want {
		t.Fatalf("go mod -packages mismatch:\nhave:<<<\n%s>>>\nwant:<<<\n%s\n", have, want)
	}

	data, err := ioutil.ReadFile(tg.path("go.mod"))
	tg.must(err)
	data = bytes.Replace(data, []byte("\n"), []byte("\r\n"), -1)
	data = append(data, "    \n"...)
	tg.must(ioutil.WriteFile(tg.path("go.mod"), data, 0666))

	tg.run("mod", "-fmt")
	mustHaveGoMod(`module x.x/y/z

exclude x.1 v1.2.0

replace x.1 => y.1/v2 v2.3.6

require x.3 v1.99.0
`)
}

func TestModLocalModule(t *testing.T) {
	// Test that local replacements work
	// and that they can use a dummy name
	// that isn't resolvable and need not even
	// include a dot. See golang.org/issue/24100.
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.must(os.MkdirAll(tg.path("x/y"), 0777))
	tg.must(os.MkdirAll(tg.path("x/z"), 0777))
	tg.must(ioutil.WriteFile(tg.path("x/y/go.mod"), []byte(`
		module x/y
		require zz v1.0.0
		replace zz v1.0.0 => ../z
	`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/y/y.go"), []byte(`package y; import _ "zz"`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/z/go.mod"), []byte(`
		module x/z
	`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/z/z.go"), []byte(`package z`), 0666))
	tg.cd(tg.path("x/y"))
	tg.run("build")
}

func TestModTags(t *testing.T) {
	// Test that build tags are used. See golang.org/issue/24053.
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.must(os.MkdirAll(tg.path("x"), 0777))
	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte(`
		module x
	`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/x.go"), []byte(`// +build tag1

		package y
	`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/y.go"), []byte(`// +build tag2

		package y
	`), 0666))
	tg.cd(tg.path("x"))

	tg.runFail("list", "-f={{.GoFiles}}")
	tg.grepStderr("build constraints exclude all Go files", "no Go source files without tags")

	tg.run("list", "-f={{.GoFiles}}", "-tags=tag1")
	tg.grepStdout(`\[x.go\]`, "Go source files for tag1")

	tg.run("list", "-f={{.GoFiles}}", "-tags", "tag2")
	tg.grepStdout(`\[y.go\]`, "Go source files for tag2")

	tg.run("list", "-f={{.GoFiles}}", "-tags", "tag1 tag2")
	tg.grepStdout(`\[x.go y.go\]`, "Go source files for tag1 and tag2")
}

func TestModFSPatterns(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.must(os.MkdirAll(tg.path("x/vendor/v"), 0777))
	tg.must(os.MkdirAll(tg.path("x/y/z/w"), 0777))
	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte(`
		module m
	`), 0666))

	tg.must(ioutil.WriteFile(tg.path("x/x.go"), []byte(`package x`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/vendor/v/v.go"), []byte(`package v; import "golang.org/x/crypto"`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/vendor/v.go"), []byte(`package main`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/y/y.go"), []byte(`package y`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/y/z/go.mod"), []byte(`syntax error`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/y/z/z.go"), []byte(`package z`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/y/z/w/w.go"), []byte(`package w`), 0666))

	tg.cd(tg.path("x"))
	for _, pattern := range []string{"all", "m/...", "./..."} {
		tg.run("list", pattern)
		tg.grepStdout(`^m$`, "expected m")
		tg.grepStdout(`^m/vendor$`, "must see package named vendor")
		tg.grepStdoutNot(`vendor/`, "must not see vendored packages")
		tg.grepStdout(`^m/y$`, "expected m/y")
		tg.grepStdoutNot(`^m/y/z`, "should ignore submodule m/y/z...")
	}
}

func TestModGetVersions(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.setenv(homeEnvName(), tg.path("home"))
	tg.must(os.MkdirAll(tg.path("x"), 0777))
	tg.cd(tg.path("x"))
	tg.must(ioutil.WriteFile(tg.path("x/x.go"), []byte(`package x`), 0666))

	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte(`
		module x
		require rsc.io/quote v1.1.0
	`), 0666))
	tg.run("get", "rsc.io/quote@v2.0.0")
	tg.run("list", "-m", "all")
	tg.grepStdout("rsc.io/quote.*v0.0.0-20180709153244-fd906ed3b100", "did downgrade to v0.0.0-*")

	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte(`
		module x
		require rsc.io/quote v1.2.0
	`), 0666))
	tg.run("get", "rsc.io/quote@v1.1.0")
	tg.run("list", "-m", "-u", "all")
	tg.grepStdout(`rsc.io/quote v1.1.0`, "did downgrade to v1.1.0")
	tg.grepStdout(`rsc.io/quote v1.1.0 \[v1`, "did show upgrade to v1.2.0 or later")

	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte(`
		module x
		require rsc.io/quote v1.1.0
	`), 0666))
	tg.run("get", "rsc.io/quote@v1.2.0")
	tg.run("list", "-m", "all")
	tg.grepStdout("rsc.io/quote.*v1.2.0", "did upgrade to v1.2.0")

	// @14c0d48ead0c should resolve,
	// and also there should be no build error about not having Go files in the root.
	tg.run("get", "golang.org/x/text@14c0d48ead0c")

	// @14c0d48ead0c should resolve.
	// Now there should be no build at all.
	tg.run("get", "-m", "golang.org/x/text@14c0d48ead0c")

	// language@7f39a6fea4fe9364 should not resolve with -m,
	// because .../language is not a module path.
	tg.runFail("get", "-m", "golang.org/x/text/language@14c0d48ead0c")

	// language@7f39a6fea4fe9364 should resolve without -m.
	// Because of -d, there should be no build at all.
	tg.run("get", "-d", "-x", "golang.org/x/text/language@14c0d48ead0c")
	tg.grepStderrNot(`compile|cp .*language\.a$`, "should not see compile steps")

	// Dropping -d, we should see a build now.
	tg.run("get", "-x", "golang.org/x/text/language@14c0d48ead0c")
	tg.grepStderr(`compile|cp .*language\.a$`, "should see compile steps")

	// Even with -d, we should see an error for unknown packages.
	tg.runFail("get", "-x", "golang.org/x/text/foo@14c0d48ead0c")
}

func TestModGetUpgrade(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.setenv(homeEnvName(), tg.path("home"))
	tg.must(os.MkdirAll(tg.path("x"), 0777))
	tg.cd(tg.path("x"))
	tg.must(ioutil.WriteFile(tg.path("x/x.go"), []byte(`package x; import _ "rsc.io/quote"`), 0666))

	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte(`
		module x
		require rsc.io/quote v1.5.1
	`), 0666))

	tg.run("get", "-x", "-u")
	tg.run("list", "-m", "-f={{.Path}} {{.Version}}{{if .Indirect}} // indirect{{end}}", "all")
	tg.grepStdout(`quote v1.5.2$`, "should have upgraded only to v1.5.2")
	tg.grepStdout(`x/text [v0-9a-f.\-]+ // indirect`, "should list golang.org/x/text as indirect")

	var gomod string
	readGoMod := func() {
		data, err := ioutil.ReadFile(tg.path("x/go.mod"))
		if err != nil {
			t.Fatal(err)
		}
		gomod = string(data)
	}
	readGoMod()
	if !strings.Contains(gomod, "rsc.io/quote v1.5.2\n") {
		t.Fatalf("expected rsc.io/quote direct requirement:\n%s", gomod)
	}
	if !regexp.MustCompile(`(?m)golang.org/x/text.* // indirect`).MatchString(gomod) {
		t.Fatalf("expected golang.org/x/text indirect requirement:\n%s", gomod)
	}

	tg.must(ioutil.WriteFile(tg.path("x/x.go"), []byte(`package x; import _ "golang.org/x/text"`), 0666))
	tg.run("list") // rescans directory
	readGoMod()
	if !strings.Contains(gomod, "rsc.io/quote v1.5.2\n") {
		t.Fatalf("expected rsc.io/quote direct requirement:\n%s", gomod)
	}
	if !regexp.MustCompile(`(?m)golang.org/x/text[^/]+\n`).MatchString(gomod) {
		t.Fatalf("expected golang.org/x/text DIRECT requirement:\n%s", gomod)
	}

	tg.must(ioutil.WriteFile(tg.path("x/x.go"), []byte(`package x; import _ "rsc.io/quote"`), 0666))
	tg.run("mod", "-sync") // rescans everything, can put // indirect marks back
	readGoMod()
	if !strings.Contains(gomod, "rsc.io/quote v1.5.2\n") {
		t.Fatalf("expected rsc.io/quote direct requirement:\n%s", gomod)
	}
	if !regexp.MustCompile(`(?m)golang.org/x/text.* // indirect\n`).MatchString(gomod) {
		t.Fatalf("expected golang.org/x/text indirect requirement:\n%s", gomod)
	}

	tg.run("get", "rsc.io/quote@v0.0.0-20180214005840-23179ee8a569") // should record as (time-corrected) pseudo-version
	readGoMod()
	if !strings.Contains(gomod, "rsc.io/quote v0.0.0-20180214005840-23179ee8a569\n") {
		t.Fatalf("expected rsc.io/quote v0.0.0-20180214005840-23179ee8a569 (not v1.5.1)\n%s", gomod)
	}

	tg.run("get", "rsc.io/quote@23179ee") // should record as v1.5.1
	readGoMod()
	if !strings.Contains(gomod, "rsc.io/quote v1.5.1\n") {
		t.Fatalf("expected rsc.io/quote v1.5.1 (not 23179ee)\n%s", gomod)
	}

	tg.run("mod", "-require", "rsc.io/quote@23179ee") // should record as 23179ee
	readGoMod()
	if !strings.Contains(gomod, "rsc.io/quote 23179ee\n") {
		t.Fatalf("expected rsc.io/quote 23179ee\n%s", gomod)
	}

	tg.run("mod", "-fix") // fixup in any future go command should find v1.5.1 again
	readGoMod()
	if !strings.Contains(gomod, "rsc.io/quote v1.5.1\n") {
		t.Fatalf("expected rsc.io/quote v1.5.1\n%s", gomod)
	}

	tg.run("get", "-m", "rsc.io/quote@dd9747d")
	tg.run("list", "-m", "all")
	tg.grepStdout(`quote v0.0.0-20180628003336-dd9747d19b04$`, "should have moved to pseudo-commit")

	tg.run("get", "-m", "-u")
	tg.run("list", "-m", "all")
	tg.grepStdout(`quote v0.0.0-20180628003336-dd9747d19b04$`, "should have stayed on pseudo-commit")

	tg.run("get", "-m", "rsc.io/quote@e7a685a342")
	tg.run("list", "-m", "all")
	tg.grepStdout(`quote v0.0.0-20180214005133-e7a685a342c0$`, "should have moved to new pseudo-commit")

	tg.run("get", "-m", "-u")
	tg.run("list", "-m", "all")
	tg.grepStdout(`quote v1.5.2$`, "should have moved off pseudo-commit")

	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte(`
		module x
	`), 0666))
	tg.run("list")
	tg.run("list", "-m", "all")
	tg.grepStdout(`quote v1.5.2$`, "should have added quote v1.5.2")
	tg.grepStdoutNot(`v1.5.3-pre1`, "should not mention v1.5.3-pre1")

	tg.run("list", "-m", "-versions", "rsc.io/quote")
	want := "rsc.io/quote v1.0.0 v1.1.0 v1.2.0 v1.2.1 v1.3.0 v1.4.0 v1.5.0 v1.5.1 v1.5.2 v1.5.3-pre1\n"
	if tg.getStdout() != want {
		t.Errorf("go list versions:\nhave:\n%s\nwant:\n%s", tg.getStdout(), want)
	}

	tg.run("list", "-m", "rsc.io/quote@>v1.5.2")
	tg.grepStdout(`v1.5.3-pre1`, "expected to find v1.5.3-pre1")
	tg.run("list", "-m", "rsc.io/quote@<v1.5.4")
	tg.grepStdout(`v1.5.2$`, "expected to find v1.5.2 (NOT v1.5.3-pre1)")

	tg.runFail("list", "-m", "rsc.io/quote@>v1.5.3")
	tg.grepStderr(`go list -m rsc.io/quote: no matching versions for query ">v1.5.3"`, "expected no matching version")

	tg.run("list", "-m", "-e", "-f={{.Error.Err}}", "rsc.io/quote@>v1.5.3")
	tg.grepStdout(`no matching versions for query ">v1.5.3"`, "expected no matching version")

	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte(`
		module x
		require rsc.io/quote v1.4.0
	`), 0666))

	tg.run("list", "-m", "all")
	tg.grepStdout(`rsc.io/sampler v1.0.0`, "expected sampler v1.0.0")

	tg.run("get", "-m", "-u=patch", "rsc.io/quote")
	tg.run("list", "-m", "all")
	tg.grepStdout(`rsc.io/quote v1.5.2`, "expected quote v1.5.2")                // rsc.io/quote gets implicit @latest (not -u=patch)
	tg.grepStdout(`rsc.io/sampler v1.3.1`, "expected sampler v1.3.1")            // even though v1.5.2 requires v1.3.0
	tg.grepStdout(`golang.org/x/text v0.0.0-`, "expected x/text pseudo-version") // can't jump from v0.0.0- to v0.3.0

	tg.run("get", "-m", "-u=patch", "rsc.io/quote@v1.2.0")
	tg.run("list", "-m", "all")
	tg.grepStdout(`rsc.io/quote v1.2.0`, "expected quote v1.2.0")           // not v1.2.1: -u=patch applies to deps of args, not args
	tg.grepStdout(`rsc.io/sampler v1.3.1`, "expected sampler line to stay") // even though v1.2.0 does not require sampler?

	tg.run("get", "-m", "-u=patch")
	tg.run("list", "-m", "all")
	tg.grepStdout(`rsc.io/quote v1.2.1`, "expected quote v1.2.1") // -u=patch with no args applies to deps of main module
	tg.grepStdout(`rsc.io/sampler v1.3.1`, "expected sampler line to stay")
	tg.grepStdout(`golang.org/x/text v0.0.0-`, "expected x/text pseudo-version") // even though x/text v0.3.0 is tagged

	tg.run("get", "-m", "rsc.io/quote@v1.5.1")
	tg.run("mod", "-vendor")
	tg.setenv("GOPATH", tg.path("empty"))
	tg.setenv("GOPROXY", "file:///nonexist")

	tg.run("list", "-getmode=vendor", "all")
	tg.run("list", "-getmode=vendor", "-m", "-f={{.Path}} {{.Version}} {{.Dir}}", "all")
	tg.grepStdout(`rsc.io/quote v1.5.1 .*vendor[\\/]rsc.io[\\/]quote`, "expected vendored rsc.io/quote")
	tg.grepStdout(`golang.org/x/text v0.0.0.* .*vendor[\\/]golang.org[\\/]x[\\/]text`, "expected vendored golang.org/x/text")

	tg.runFail("list", "-getmode=vendor", "-m", "rsc.io/quote@latest")
	tg.grepStderr(`module lookup disabled by -getmode=vendor`, "expected disabled")
	tg.runFail("get", "-getmode=vendor", "-u")
	tg.grepStderr(`go get: disabled by -getmode=vendor`, "expected disabled")
}

func TestModBadDomain(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.tempFile("work/x.go", `
		package x

		import _ "appengine"
		import _ "nonexistent.rsc.io" // domain does not exist
	`)

	tg.runFail("get", "appengine")
	tg.grepStderr(`cannot find module providing package appengine`, "expected module error ")
	tg.runFail("get", "x/y.z")
	tg.grepStderr(`cannot find module providing package x/y.z`, "expected module error")

	tg.runFail("build")
	tg.grepStderrNot("unknown module appengine: not a domain name", "expected nothing about appengine")
	tg.grepStderr("cannot find module providing package nonexistent.rsc.io", "expected error for nonexistent.rsc.io")
}

func TestModSync(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	write := func(name, text string) {
		name = tg.path(name)
		dir := filepath.Dir(name)
		tg.must(os.MkdirAll(dir, 0777))
		tg.must(ioutil.WriteFile(name, []byte(text), 0666))
	}

	write("m/go.mod", `
module m

require (
	x.1 v1.0.0
	y.1 v1.0.0
	w.1 v1.2.0
)

replace x.1 v1.0.0 => ../x
replace y.1 v1.0.0 => ../y
replace z.1 v1.1.0 => ../z
replace z.1 v1.2.0 => ../z
replace w.1 => ../w
`)
	write("m/m.go", `
package m

import _ "x.1"
import _ "z.1/sub"
`)

	write("w/go.mod", `
module w
`)
	write("w/w.go", `
package w
`)

	write("x/go.mod", `
module x
require w.1 v1.1.0
require z.1 v1.1.0
`)
	write("x/x.go", `
package x

import _ "w.1"
`)

	write("y/go.mod", `
module y
require z.1 v1.2.0
`)

	write("z/go.mod", `
module z
`)
	write("z/sub/sub.go", `
package sub
`)

	tg.cd(tg.path("m"))
	tg.run("mod", "-sync", "-v")
	tg.grepStderr(`^unused y.1`, "need y.1 unused")
	tg.grepStderrNot(`^unused [^y]`, "only y.1 should be unused")

	tg.run("list", "-m", "all")
	tg.grepStdoutNot(`^y.1`, "y should be gone")
	tg.grepStdout(`^w.1\s+v1.2.0`, "need w.1 to stay at v1.2.0")
	tg.grepStdout(`^z.1\s+v1.2.0`, "need z.1 to stay at v1.2.0 even though y is gone")
}

func TestModVendor(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.extract("testdata/vendormod.txt")

	tg.run("list", "-m", "all")
	tg.grepStdout(`^x`, "expected to see module x")
	tg.grepStdout(`=> ./x`, "expected to see replacement for module x")
	tg.grepStdout(`^w`, "expected to see module w")

	if !testing.Short() {
		tg.run("build")
		tg.runFail("build", "-getmode=vendor")
	}

	tg.run("list", "-f={{.Dir}}", "x")
	tg.grepStdout(`work[/\\]x$`, "expected x in work/x")

	mustHaveVendor := func(name string) {
		t.Helper()
		tg.mustExist(filepath.Join(tg.path("work/vendor"), name))
	}
	mustNotHaveVendor := func(name string) {
		t.Helper()
		tg.mustNotExist(filepath.Join(tg.path("work/vendor"), name))
	}

	tg.run("mod", "-vendor", "-v")
	tg.grepStderr(`^# x v1.0.0 => ./x`, "expected to see module x with replacement")
	tg.grepStderr(`^x`, "expected to see package x")
	tg.grepStderr(`^# y v1.0.0 => ./y`, "expected to see module y with replacement")
	tg.grepStderr(`^y`, "expected to see package y")
	tg.grepStderr(`^# z v1.0.0 => ./z`, "expected to see module z with replacement")
	tg.grepStderr(`^z`, "expected to see package z")
	tg.grepStderrNot(`w`, "expected NOT to see unused module w")

	tg.run("list", "-f={{.Dir}}", "x")
	tg.grepStdout(`work[/\\]x$`, "expected x in work/x")

	tg.run("list", "-f={{.Dir}}", "-m", "x")
	tg.grepStdout(`work[/\\]x$`, "expected x in work/x")

	tg.run("list", "-getmode=vendor", "-f={{.Dir}}", "x")
	tg.grepStdout(`work[/\\]vendor[/\\]x$`, "expected x in work/vendor/x in -get=vendor mode")

	tg.run("list", "-getmode=vendor", "-f={{.Dir}}", "-m", "x")
	tg.grepStdout(`work[/\\]vendor[/\\]x$`, "expected x in work/vendor/x in -get=vendor mode")

	tg.run("list", "-f={{.Dir}}", "w")
	tg.grepStdout(`work[/\\]w$`, "expected w in work/w")
	tg.runFail("list", "-getmode=vendor", "-f={{.Dir}}", "w")
	tg.grepStderr(`work[/\\]vendor[/\\]w`, "want error about work/vendor/w not existing")

	tg.run("list", "-getmode=local", "-f={{.Dir}}", "w")
	tg.grepStdout(`work[/\\]w`, "expected w in work/w")

	tg.runFail("list", "-getmode=local", "-f={{.Dir}}", "newpkg")
	tg.grepStderr(`disabled by -getmode=local`, "expected -getmode=local to avoid network")

	mustNotHaveVendor("x/testdata")
	mustNotHaveVendor("a/foo/bar/b/main_test.go")

	mustHaveVendor("a/foo/AUTHORS.txt")
	mustHaveVendor("a/foo/CONTRIBUTORS")
	mustHaveVendor("a/foo/LICENSE")
	mustHaveVendor("a/foo/PATENTS")
	mustHaveVendor("a/foo/COPYING")
	mustHaveVendor("a/foo/COPYLEFT")
	mustHaveVendor("x/NOTICE!")
	mustHaveVendor("mysite/myname/mypkg/LICENSE.txt")

	mustNotHaveVendor("a/foo/licensed-to-kill")
	mustNotHaveVendor("w")
	mustNotHaveVendor("w/LICENSE") // w wasn't copied at all
	mustNotHaveVendor("x/x2")
	mustNotHaveVendor("x/x2/LICENSE") // x/x2 wasn't copied at all

	if !testing.Short() {
		tg.run("build")
		tg.run("build", "-getmode=vendor")
		tg.run("test", "-getmode=vendor", ".", "./subdir")
		tg.run("test", "-getmode=vendor", "./...")
	}
}

func TestModList(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.setenv(homeEnvName(), tg.path("."))
	tg.must(os.MkdirAll(tg.path("x"), 0777))
	tg.must(ioutil.WriteFile(tg.path("x/x.go"), []byte(`
		package x
		import _ "rsc.io/quote"
	`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte(`
		module x
		require rsc.io/quote v1.5.1
		replace rsc.io/sampler v1.3.0 => rsc.io/sampler v1.3.1
	`), 0666))
	tg.cd(tg.path("x"))

	tg.run("list", "-m", "-f={{.Main}}: {{.Dir}}")
	tg.grepStdout(`^true: `, "expected main module to have Main=true")
	tg.grepStdout(regexp.QuoteMeta(tg.path("x")), "expected Dir of main module to be present")

	tg.run("list", "-m", "-f={{.Main}}: {{.Dir}}", "rsc.io/quote")
	tg.grepStdout(`^false: `, "expected non-main module to have Main=false")
	tg.grepStdoutNot(`quote@`, "should not have local copy of code")

	tg.run("list", "-f={{.Dir}}", "rsc.io/quote") // downloads code to load package
	tg.grepStdout(`mod[\\/]rsc.io[\\/]quote@v1.5.1`, "expected cached copy of code")
	dir := strings.TrimSpace(tg.getStdout())
	info, err := os.Stat(dir)
	if err != nil {
		t.Fatal(err)
	}
	if info.Mode()&0222 != 0 {
		t.Fatalf("%s should be unwritable", dir)
	}

	tg.run("list", "-m", "-f={{.Path}} {{.Version}} {{.Dir}}{{with .Replace}} => {{.Version}} {{.Dir}}{{end}}", "all")
	tg.grepStdout(`mod[\\/]rsc.io[\\/]quote@v1.5.1`, "expected cached copy of code")
	tg.grepStdout(`v1.3.0 .*mod[\\/]rsc.io[\\/]sampler@v1.3.1 => v1.3.1 .*@v1.3.1`, "expected v1.3.1 replacement")

	// check that list std works; also check that rsc.io/quote/buggy is a listable package
	tg.run("list", "std", "rsc.io/quote/buggy")
	tg.grepStdout("^math/big", "expected standard library")

	tg.run("list", "-m", "-e", "-f={{.Path}} {{.Error.Err}}", "nonexist", "rsc.io/quote/buggy")
	tg.grepStdout(`^nonexist module "nonexist" is not a known dependency`, "expected error via template")
	tg.grepStdout(`^rsc.io/quote/buggy module "rsc.io/quote/buggy" is not a known dependency`, "expected error via template")

	tg.runFail("list", "-m", "nonexist", "rsc.io/quote/buggy")
	tg.grepStderr(`go list -m nonexist: module "nonexist" is not a known dependency`, "expected error on stderr")
	tg.grepStderr(`go list -m rsc.io/quote/buggy: module "rsc.io/quote/buggy" is not a known dependency`, "expected error on stderr")

	// Check that module loader does not interfere with list -e (golang.org/issue/24149).
	tg.run("list", "-e", "-f={{.ImportPath}} {{.Error.Err}}", "database")
	tg.grepStdout(`^database no Go files in `, "expected error via template")
	tg.runFail("list", "database")
	tg.grepStderr(`package database: no Go files`, "expected error on stderr")

}

func TestModInitLegacy(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("skipping because git binary not found")
	}

	tg := testGoModules(t)
	defer tg.cleanup()

	tg.setenv(homeEnvName(), tg.path("."))
	tg.must(os.MkdirAll(tg.path("x"), 0777))
	tg.must(ioutil.WriteFile(tg.path("x/x.go"), []byte(`
		package x
	`), 0666))

	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte(`
		module x
	`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/Gopkg.lock"), []byte(`
[[projects]]
  name = "rsc.io/sampler"
  version = "v1.0.0"
	`), 0666))

	tg.cd(tg.path("x"))
	tg.run("build", "-v")
	tg.grepStderr("copying requirements from .*Gopkg.lock", "did not copy Gopkg.lock")
	tg.run("list", "-m", "all")
	tg.grepStderrNot("copying requirements from .*Gopkg.lock", "should not copy Gopkg.lock again")
	tg.grepStdout("rsc.io/sampler.*v1.0.0", "did not copy Gopkg.lock")

	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte(`
		module x
	`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/Gopkg.lock"), []byte(`
	`), 0666))

	tg.run("list")
	tg.grepStderr("copying requirements from .*Gopkg.lock", "did not copy Gopkg.lock")
	tg.run("list")
	tg.grepStderrNot("copying requirements from .*Gopkg.lock", "should not copy Gopkg.lock again")
}

func TestModQueryExcluded(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.must(os.MkdirAll(tg.path("x"), 0777))
	tg.must(ioutil.WriteFile(tg.path("x/x.go"), []byte(`package x; import _ "github.com/gorilla/mux"`), 0666))
	gomod := []byte(`
		module x

		exclude rsc.io/quote v1.5.0
	`)

	tg.setenv(homeEnvName(), tg.path("home"))
	tg.cd(tg.path("x"))

	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), gomod, 0666))
	tg.runFail("get", "rsc.io/quote@v1.5.0")
	tg.grepStderr("rsc.io/quote@v1.5.0 excluded", "print version excluded")

	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), gomod, 0666))
	tg.run("get", "rsc.io/quote@v1.5.1")
	tg.grepStderr("rsc.io/quote v1.5.1", "find version 1.5.1")

	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), gomod, 0666))
	tg.run("get", "rsc.io/quote@>=v1.5")
	tg.run("list", "-m", "...quote")
	tg.grepStdout("rsc.io/quote v1.5.[1-9]", "expected version 1.5.1 or later")
}

func TestModRequireExcluded(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.must(os.MkdirAll(tg.path("x"), 0777))
	tg.must(ioutil.WriteFile(tg.path("x/x.go"), []byte(`package x; import _ "rsc.io/quote"`), 0666))

	tg.setenv(homeEnvName(), tg.path("home"))
	tg.cd(tg.path("x"))

	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte(`
		module x
		exclude rsc.io/sampler latest
		require rsc.io/sampler latest
	`), 0666))
	tg.runFail("build")
	tg.grepStderr("no newer version available", "only available version excluded")

	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte(`
		module x
		exclude rsc.io/quote v1.5.1
		require rsc.io/quote v1.5.1
	`), 0666))
	tg.run("build")
	tg.grepStderr("rsc.io/quote v1.5.2", "find version 1.5.2")

	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte(`
		module x
		exclude rsc.io/quote v1.5.2
		require rsc.io/quote v1.5.1
	`), 0666))
	tg.run("build")
	tg.grepStderr("rsc.io/quote v1.5.1", "find version 1.5.1")
}

func TestModInitLegacy2(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("skipping because git binary not found")
	}

	tg := testGoModules(t)
	defer tg.cleanup()

	tg.setenv(homeEnvName(), tg.path("."))

	// Testing that on Windows the path x/Gopkg.lock turning into x\Gopkg.lock does not confuse converter.
	tg.must(os.MkdirAll(tg.path("x"), 0777))
	tg.must(ioutil.WriteFile(tg.path("x/Gopkg.lock"), []byte(`
	  [[projects]]
		name = "rsc.io/quote"
		packages = ["."]
		revision = "645ef00459ed84a119197bfb8d8205042c6df63d"
		version = "v1.4.0"`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/main.go"), []byte("package x // import \"x\"\n import _ \"github.com/pkg/errors\""), 0666))
	tg.cd(tg.path("x"))
	tg.run("list", "-m", "all")

	// If the conversion just ignored the Gopkg.lock entirely
	// it would choose a newer version (like v1.5.2 or maybe
	// something even newer). Check for the older version to
	// make sure Gopkg.lock was properly used.
	tg.grepStdout("v1.4.0", "expected rsc.io/quote at v1.4.0")
}

func TestModVerify(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	gopath := tg.path("gp")
	tg.setenv("GOPATH", gopath)
	tg.must(os.MkdirAll(tg.path("x"), 0777))
	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte(`
		module x
		require rsc.io/quote v1.1.0
	`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/x.go"), []byte(`package x; import _ "rsc.io/quote"`), 0666))

	// With correct go.sum,verify succeeds but avoids download.
	tg.must(ioutil.WriteFile(tg.path("x/go.sum"), []byte(`rsc.io/quote v1.1.0 h1:WdK/asTD0HN+q6hsWO3/vpuAkAr+tw6aNJNDFFf0+qw=
`), 0666))
	tg.cd(tg.path("x"))
	tg.run("mod", "-verify")
	tg.mustNotExist(filepath.Join(gopath, "src/mod/cache/download/rsc.io/quote/@v/v1.1.0.zip"))
	tg.mustNotExist(filepath.Join(gopath, "src/mod/github.com/pkg"))

	// With incorrect sum, sync (which must download) fails.
	// Even if the incorrect sum is in the old legacy go.modverify file.
	tg.must(ioutil.WriteFile(tg.path("x/go.sum"), []byte(`
`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/go.modverify"), []byte(`rsc.io/quote v1.1.0 h1:a3YaZoizPtXyv6ZsJ74oo2L4/bwOSTKMY7MAyo4O/1c=
`), 0666))
	tg.runFail("mod", "-sync") // downloads pkg/errors
	tg.grepStderr("checksum mismatch", "must detect mismatch")
	tg.mustNotExist(filepath.Join(gopath, "src/mod/cache/download/rsc.io/quote/@v/v1.1.0.zip"))
	tg.mustNotExist(filepath.Join(gopath, "src/mod/github.com/pkg"))

	// With corrected sum, sync works.
	tg.must(ioutil.WriteFile(tg.path("x/go.modverify"), []byte(`rsc.io/quote v1.1.0 h1:a3YaZoizPtXyv6ZsJ74oo2L4/bwOSTKMY7MAyo4O/0c=
`), 0666))
	tg.run("mod", "-sync")
	tg.mustExist(filepath.Join(gopath, "src/mod/cache/download/rsc.io/quote/@v/v1.1.0.zip"))
	tg.mustExist(filepath.Join(gopath, "src/mod/rsc.io"))
	tg.mustNotExist(tg.path("x/go.modverify")) // moved into go.sum

	// Sync should have added sum for go.mod.
	data, err := ioutil.ReadFile(tg.path("x/go.sum"))
	if !strings.Contains(string(data), "\nrsc.io/quote v1.1.0/go.mod ") {
		t.Fatalf("cannot find go.mod hash in go.sum: %v\n%s", err, data)
	}

	// Verify should work too.
	tg.run("mod", "-verify")

	// Even the most basic attempt to load the module graph should detect incorrect go.mod files.
	tg.run("mod", "-graph") // loads module graph, is OK
	tg.must(ioutil.WriteFile(tg.path("x/go.sum"), []byte(`rsc.io/quote v1.1.0 a3YaZoizPtXyv6ZsJ74oo2L4/bwOSTKMY7MAyo4O/1c=
rsc.io/quote v1.1.0/go.mod h1:bwawxfHBFNV+L2hUp1rHADufV3IMtnDRdf1r5NINEl1=
`), 0666))
	tg.runFail("mod", "-graph") // loads module graph, fails (even though sum is in old go.modverify file)
	tg.grepStderr("go.mod: checksum mismatch", "must detect mismatch")

	// go.sum should be created and updated automatically.
	tg.must(os.Remove(tg.path("x/go.sum")))
	tg.run("mod", "-graph")
	tg.mustExist(tg.path("x/go.sum"))
	data, err = ioutil.ReadFile(tg.path("x/go.sum"))
	if !strings.Contains(string(data), " v1.1.0/go.mod ") {
		t.Fatalf("cannot find go.mod hash in go.sum: %v\n%s", err, data)
	}
	if strings.Contains(string(data), " v1.1.0 ") {
		t.Fatalf("unexpected module tree hash in go.sum: %v\n%s", err, data)
	}
	tg.run("mod", "-sync")
	data, err = ioutil.ReadFile(tg.path("x/go.sum"))
	if !strings.Contains(string(data), " v1.1.0/go.mod ") {
		t.Fatalf("cannot find go.mod hash in go.sum: %v\n%s", err, data)
	}
	if !strings.Contains(string(data), " v1.1.0 ") {
		t.Fatalf("cannot find module tree hash in go.sum: %v\n%s", err, data)
	}

	tg.must(os.Remove(filepath.Join(gopath, "src/mod/cache/download/rsc.io/quote/@v/v1.1.0.ziphash")))
	tg.run("mod", "-sync") // ignores missing ziphash file for ordinary go.sum validation

	tg.runFail("mod", "-verify") // explicit verify fails with missing ziphash

	tg.run("mod", "-droprequire", "rsc.io/quote")
	tg.run("list", "rsc.io/quote/buggy")
	data, err = ioutil.ReadFile(tg.path("x/go.sum"))
	if strings.Contains(string(data), "buggy") {
		t.Fatalf("did not expect buggy in go.sum:\n%s", data)
	}
	if !strings.Contains(string(data), "rsc.io/quote v1.5.2/go.mod") {
		t.Fatalf("did expect rsc.io/quote go.mod in go.sum:\n%s", data)
	}

	tg.run("mod", "-droprequire", "rsc.io/quote")
	tg.runFail("list", "rsc.io/quote/buggy/foo")
	data, err = ioutil.ReadFile(tg.path("x/go.sum"))
	if strings.Contains(string(data), "buggy") {
		t.Fatalf("did not expect buggy in go.sum:\n%s", data)
	}
	if !strings.Contains(string(data), "rsc.io/quote v1.5.2/go.mod") {
		t.Fatalf("did expect rsc.io/quote go.mod in go.sum:\n%s", data)
	}

	tg.run("mod", "-droprequire", "rsc.io/quote")
	tg.runFail("list", "rsc.io/quote/morebuggy")
	if strings.Contains(string(data), "morebuggy") {
		t.Fatalf("did not expect morebuggy in go.sum:\n%s", data)
	}
	if !strings.Contains(string(data), "rsc.io/quote v1.5.2/go.mod") {
		t.Fatalf("did expect rsc.io/quote go.mod in go.sum:\n%s", data)
	}
}

func TestModFileProxy(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.setenv("GOPATH", tg.path("gp1"))

	tg.must(os.MkdirAll(tg.path("x"), 0777))
	tg.must(ioutil.WriteFile(tg.path("x/main.go"), []byte(`package x; import _ "rsc.io/quote"`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte(`module x
		require rsc.io/quote v1.5.1`), 0666))
	tg.cd(tg.path("x"))
	tg.run("list", "all")
	tg.run("list", "-getmode=local", "all")
	tg.mustExist(tg.path("gp1/src/mod/cache/download/rsc.io/quote/@v/list"))

	// @v/list should contain version list.
	data, err := ioutil.ReadFile(tg.path("gp1/src/mod/cache/download/rsc.io/quote/@v/list"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "v1.5.1\n") {
		t.Fatalf("cannot find v1.5.1 in @v/list:\n%s", data)
	}

	tg.setenv("GOPROXY", "file:///nonexist")
	tg.run("list", "-getmode=local", "all")

	tg.setenv("GOPATH", tg.path("gp2"))
	tg.runFail("list", "-getmode=local", "all")
	tg.runFail("list", "all") // because GOPROXY is bogus

	tg.setenv("GOPROXY", "file://"+filepath.ToSlash(tg.path("gp1/src/mod/cache/download")))
	tg.runFail("list", "-getmode=local", "all")
	tg.run("list", "all")
	tg.mustExist(tg.path("gp2/src/mod/cache/download/rsc.io/quote/@v/list"))
}

func TestModVendorNoDeps(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.must(os.MkdirAll(tg.path("x"), 0777))
	tg.must(ioutil.WriteFile(tg.path("x/main.go"), []byte(`package x`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte(`module x`), 0666))
	tg.cd(tg.path("x"))
	tg.run("mod", "-vendor")
	tg.grepStderr("go: no dependencies to vendor", "print vendor info")
}

func TestModVersionNoModule(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.cd(tg.path("."))
	tg.run("version")
}

func TestModImportDomainRoot(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.setenv("GOPATH", tg.path("."))
	tg.must(os.MkdirAll(tg.path("x"), 0777))
	tg.must(ioutil.WriteFile(tg.path("x/main.go"), []byte(`
		package x
		import _ "example.com"`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte("module x"), 0666))
	tg.cd(tg.path("x"))
	tg.run("build")
}

func TestModSyncPrintJson(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	tg.setenv("GOPATH", tg.path("."))
	tg.must(os.MkdirAll(tg.path("x"), 0777))
	tg.must(ioutil.WriteFile(tg.path("x/main.go"), []byte(`
		package x
		import "rsc.io/quote"
		func main() {
			_ = mux.NewRouter()
		}`), 0666))
	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte("module x"), 0666))
	tg.cd(tg.path("x"))
	tg.run("mod", "-sync", "-json")
	count := tg.grepCountBoth(`"Path": "rsc.io/quote",`)
	if count != 1 {
		t.Fatal("produces duplicate imports")
	}
	// test quoted module path
	tg.must(ioutil.WriteFile(tg.path("x/go.mod"), []byte(`
		module x
		require (
			"rsc.io/sampler" v1.3.0
			"rsc.io/quote" v1.5.2
	)`), 0666))
	tg.run("mod", "-sync", "-json")
	count = tg.grepCountBoth(`"Path": "rsc.io/quote",`)
	if count != 1 {
		t.Fatal("produces duplicate imports")
	}
}

func TestModMultiVersion(t *testing.T) {
	tg := testGoModules(t)
	defer tg.cleanup()

	checkModules := func(dirs ...string) {
		t.Helper()
		tg.run("list", "-deps", "-f", "{{.ImportPath}}: {{.Dir}}")
		for _, line := range strings.Split(tg.getStdout(), "\n") {
			line = strings.Replace(line, `\`, `/`, -1) // windows!
			if strings.HasPrefix(line, "rsc.io/quote: ") {
				if strings.Contains(line, "/src/mod/") {
					t.Fatalf("rsc.io/quote should not be from module cache: %v", line)
				}
			} else if strings.Contains(line, "rsc.io/quote") {
				if !strings.Contains(line, "/src/mod/") {
					t.Fatalf("rsc.io/quote/* should be from module cache: %v", line)
				}
			}
		}
	}

	tg.extract("testdata/mod/rsc.io_quote_v1.5.2.txt")
	checkModules()

	// These are git checkouts from rsc.io/quote not downlaoded modules.
	// As such they contain extra files like spurious pieces of other modules.
	tg.extract("testdata/rsc.io_quote_0d003b9.txt") // wraps v2
	checkModules()

	tg.extract("testdata/rsc.io_quote_b44a0b1.txt") // adds go.mod
	checkModules()

	tg.extract("testdata/rsc.io_quote_fe488b8.txt") // adds broken v3 subdirectory
	tg.run("list", "./...")                         // should ignore v3 because v3/go.mod exists
	checkModules()

	tg.extract("testdata/rsc.io_quote_a91498b.txt") // wraps v3
	checkModules()                                  // looks up v3 from internet, not v3 subdirectory

	tg.extract("testdata/rsc.io_quote_5d9f230.txt") // adds go.mod
	checkModules()                                  // knows which v3 to use (still needs download from internet, cached from last step)
}
