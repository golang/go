// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfile

import (
	"encoding/json"
	"flag"
	"internal/coverage"
	"internal/goexperiment"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	_ "unsafe"
)

func testGoCoverDir(t *testing.T) string {
	if f := flag.Lookup("test.gocoverdir"); f != nil {
		if dir := f.Value.String(); dir != "" {
			return dir
		}
	}
	return t.TempDir()
}

// TestTestSupport does a basic verification of the functionality in
// ProcessCoverTestDir (doing this here as opposed to
// relying on other test paths will provide a better signal when
// running "go test -cover" for this package).
func TestTestSupport(t *testing.T) {
	if !goexperiment.CoverageRedesign {
		return
	}
	if testing.CoverMode() == "" {
		return
	}
	tgcd := testGoCoverDir(t)
	t.Logf("testing.testGoCoverDir() returns %s mode=%s\n",
		tgcd, testing.CoverMode())

	textfile := filepath.Join(t.TempDir(), "file.txt")
	var sb strings.Builder
	err := ProcessCoverTestDir(tgcd, textfile,
		testing.CoverMode(), "", &sb, nil)
	if err != nil {
		t.Fatalf("bad: %v", err)
	}

	// Check for existence of text file.
	if inf, err := os.Open(textfile); err != nil {
		t.Fatalf("problems opening text file %s: %v", textfile, err)
	} else {
		inf.Close()
	}

	// Check for percent output with expected tokens.
	strout := sb.String()
	want := "of statements"
	if !strings.Contains(strout, want) {
		t.Logf("output from run: %s\n", strout)
		t.Fatalf("percent output missing token: %q", want)
	}
}

// Kicks off a sub-test to verify that Snapshot() works properly.
// We do this as a separate shell-out, so as to avoid potential
// interactions with -coverpkg. For example, if you do
//
//	$ cd `go env GOROOT`
//	$ cd src/internal/coverage
//	$ go test -coverpkg=internal/coverage/decodecounter ./...
//	...
//	$
//
// The previous version of this test could fail due to the fact
// that "cfile" itself was not being instrumented, as in the
// scenario above.
func TestCoverageSnapshot(t *testing.T) {
	testenv.MustHaveGoRun(t)
	args := []string{"test", "-tags", "SELECT_USING_THIS_TAG",
		"-cover", "-run=TestCoverageSnapshotImpl", "internal/coverage/cfile"}
	cmd := exec.Command(testenv.GoToolPath(t), args...)
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("go test failed (%v): %s", err, b)
	}
}

const hellogo = `
package main

func main() {
  println("hello")
}
`

// Returns a pair F,T where F is a meta-data file generated from
// "hello.go" above, and T is a token to look for that should be
// present in the coverage report from F.
func genAuxMeta(t *testing.T, dstdir string) (string, string) {
	// Do a GOCOVERDIR=<tmp> go run hello.go
	src := filepath.Join(dstdir, "hello.go")
	if err := os.WriteFile(src, []byte(hellogo), 0777); err != nil {
		t.Fatalf("write failed: %v", err)
	}
	args := []string{"run", "-covermode=" + testing.CoverMode(), src}
	cmd := exec.Command(testenv.GoToolPath(t), args...)
	cmd.Env = updateGoCoverDir(os.Environ(), dstdir, true)
	if b, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("go run failed (%v): %s", err, b)
	}

	// Pick out the generated meta-data file.
	files, err := os.ReadDir(dstdir)
	if err != nil {
		t.Fatalf("reading %s: %v", dstdir, err)
	}
	for _, f := range files {
		if strings.HasPrefix(f.Name(), "covmeta") {
			return filepath.Join(dstdir, f.Name()), "hello.go:"
		}
	}
	t.Fatalf("could not locate generated meta-data file")
	return "", ""
}

func TestAuxMetaDataFiles(t *testing.T) {
	if !goexperiment.CoverageRedesign {
		return
	}
	if testing.CoverMode() == "" {
		return
	}
	testenv.MustHaveGoRun(t)
	tgcd := testGoCoverDir(t)
	t.Logf("testing.testGoCoverDir() returns %s mode=%s\n",
		tgcd, testing.CoverMode())

	td := t.TempDir()

	// Manufacture a new, separate meta-data file not related to this
	// test. Contents are not important, just so long as the
	// packages/paths are different.
	othermetadir := filepath.Join(td, "othermeta")
	if err := os.Mkdir(othermetadir, 0777); err != nil {
		t.Fatalf("mkdir failed: %v", err)
	}
	mfile, token := genAuxMeta(t, othermetadir)

	// Write a metafiles file.
	metafiles := filepath.Join(tgcd, coverage.MetaFilesFileName)
	mfc := coverage.MetaFileCollection{
		ImportPaths:       []string{"command-line-arguments"},
		MetaFileFragments: []string{mfile},
	}
	jdata, err := json.Marshal(mfc)
	if err != nil {
		t.Fatalf("marshal MetaFileCollection: %v", err)
	}
	if err := os.WriteFile(metafiles, jdata, 0666); err != nil {
		t.Fatalf("write failed: %v", err)
	}

	// Kick off guts of test.
	var sb strings.Builder
	textfile := filepath.Join(td, "file2.txt")
	err = ProcessCoverTestDir(tgcd, textfile,
		testing.CoverMode(), "", &sb, nil)
	if err != nil {
		t.Fatalf("bad: %v", err)
	}
	if err = os.Remove(metafiles); err != nil {
		t.Fatalf("removing metafiles file: %v", err)
	}

	// Look for the expected things in the coverage profile.
	contents, err := os.ReadFile(textfile)
	strc := string(contents)
	if err != nil {
		t.Fatalf("problems reading text file %s: %v", textfile, err)
	}
	if !strings.Contains(strc, token) {
		t.Logf("content: %s\n", string(contents))
		t.Fatalf("cov profile does not contain aux meta content %q", token)
	}
}
