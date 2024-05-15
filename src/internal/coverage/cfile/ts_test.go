// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cfile

import (
	"encoding/json"
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

//go:linkname testing_testGoCoverDir testing.testGoCoverDir
func testing_testGoCoverDir() string

func testGoCoverDir(t *testing.T) string {
	tgcd := testing_testGoCoverDir()
	if tgcd != "" {
		return tgcd
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
		testing.CoverMode(), "", &sb)
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

var funcInvoked bool

//go:noinline
func thisFunctionOnlyCalledFromSnapshotTest(n int) int {
	if funcInvoked {
		panic("bad")
	}
	funcInvoked = true

	// Contents here not especially important, just so long as we
	// have some statements.
	t := 0
	for i := 0; i < n; i++ {
		for j := 0; j < i; j++ {
			t += i ^ j
		}
	}
	return t
}

// Tests runtime/coverage.snapshot() directly. Note that if
// coverage is not enabled, the hook is designed to just return
// zero.
func TestCoverageSnapshot(t *testing.T) {
	C1 := Snapshot()
	thisFunctionOnlyCalledFromSnapshotTest(15)
	C2 := Snapshot()
	cond := "C1 > C2"
	val := C1 > C2
	if testing.CoverMode() != "" {
		cond = "C1 >= C2"
		val = C1 >= C2
	}
	t.Logf("%f %f\n", C1, C2)
	if val {
		t.Errorf("erroneous snapshots, %s = true C1=%f C2=%f",
			cond, C1, C2)
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
		testing.CoverMode(), "", &sb)
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
