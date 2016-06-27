// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"bytes"
	"flag"
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

const (
	dataDir = "testdata"
	binary  = "testvet.exe"
)

// We implement TestMain so remove the test binary when all is done.
func TestMain(m *testing.M) {
	flag.Parse()
	result := m.Run()
	os.Remove(binary)
	os.Exit(result)
}

func MustHavePerl(t *testing.T) {
	switch runtime.GOOS {
	case "plan9", "windows":
		t.Skipf("skipping test: perl not available on %s", runtime.GOOS)
	}
	if _, err := exec.LookPath("perl"); err != nil {
		t.Skipf("skipping test: perl not found in path")
	}
}

var (
	built  = false // We have built the binary.
	failed = false // We have failed to build the binary, don't try again.
)

func Build(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	MustHavePerl(t)
	if built {
		return
	}
	if failed {
		t.Skip("cannot run on this environment")
	}
	cmd := exec.Command("go", "build", "-o", binary)
	output, err := cmd.CombinedOutput()
	if err != nil {
		failed = true
		fmt.Fprintf(os.Stderr, "%s\n", output)
		t.Fatal(err)
	}
	built = true
}

func Vet(t *testing.T, files []string) {
	errchk := filepath.Join(runtime.GOROOT(), "test", "errchk")
	flags := []string{
		"./" + binary,
		"-printfuncs=Warn:1,Warnf:1",
		"-all",
		"-shadow",
	}
	cmd := exec.Command(errchk, append(flags, files...)...)
	if !run(cmd, t) {
		t.Fatal("vet command failed")
	}
}

// Run this shell script, but do it in Go so it can be run by "go test".
// 	go build -o testvet
// 	$(GOROOT)/test/errchk ./testvet -shadow -printfuncs='Warn:1,Warnf:1' testdata/*.go testdata/*.s
// 	rm testvet
//

func TestVet(t *testing.T) {
	Build(t)

	// errchk ./testvet
	gos, err := filepath.Glob(filepath.Join(dataDir, "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	asms, err := filepath.Glob(filepath.Join(dataDir, "*.s"))
	if err != nil {
		t.Fatal(err)
	}
	files := append(gos, asms...)
	Vet(t, files)
}

func TestDivergentPackagesExamples(t *testing.T) {
	Build(t)
	// errchk ./testvet
	Vet(t, []string{"testdata/divergent"})
}

func TestIncompleteExamples(t *testing.T) {
	Build(t)
	// errchk ./testvet
	Vet(t, []string{"testdata/incomplete/examples_test.go"})
}

func run(c *exec.Cmd, t *testing.T) bool {
	output, err := c.CombinedOutput()
	os.Stderr.Write(output)
	if err != nil {
		t.Fatal(err)
	}
	// Errchk delights by not returning non-zero status if it finds errors, so we look at the output.
	// It prints "BUG" if there is a failure.
	if !c.ProcessState.Success() {
		return false
	}
	return !bytes.Contains(output, []byte("BUG"))
}

// TestTags verifies that the -tags argument controls which files to check.
func TestTags(t *testing.T) {
	Build(t)
	args := []string{
		"-tags=testtag",
		"-v", // We're going to look at the files it examines.
		"testdata/tagtest",
	}
	cmd := exec.Command("./"+binary, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}
	// file1 has testtag and file2 has !testtag.
	if !bytes.Contains(output, []byte(filepath.Join("tagtest", "file1.go"))) {
		t.Error("file1 was excluded, should be included")
	}
	if bytes.Contains(output, []byte(filepath.Join("tagtest", "file2.go"))) {
		t.Error("file2 was included, should be excluded")
	}
}
