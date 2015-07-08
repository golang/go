// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// No testdata on Android.

// +build !android

package main_test

import (
	"bytes"
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

func CanRun(t *testing.T) bool {
	// Plan 9 and Windows systems can't be guaranteed to have Perl and so can't run errchk.
	switch runtime.GOOS {
	case "plan9", "windows":
		t.Skip("skipping test; no Perl on %q", runtime.GOOS)
		return false
	}
	return true
}

func Build(t *testing.T) {
	// go build
	cmd := exec.Command("go", "build", "-o", binary)
	run(cmd, t)
}

func Vet(t *testing.T, files []string) {
	errchk := filepath.Join(runtime.GOROOT(), "test", "errchk")
	flags := []string{
		"./" + binary,
		"-printfuncs=Warn:1,Warnf:1",
		"-test", // TODO: Delete once -shadow is part of -all.
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
	if !CanRun(t) {
		t.Skip("cannot run on this environment")
	}
	Build(t)
	defer os.Remove(binary)

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
	if !CanRun(t) {
		t.Skip("cannot run on this environment")
	}
	Build(t)
	defer os.Remove(binary)

	// errchk ./testvet
	Vet(t, []string{"testdata/divergent/buf.go", "testdata/divergent/buf_test.go"})
}

func TestIncompleteExamples(t *testing.T) {
	if !CanRun(t) {
		t.Skip("cannot run on this environment")
	}
	Build(t)
	defer os.Remove(binary)

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
	// go build
	cmd := exec.Command("go", "build", "-o", binary)
	run(cmd, t)

	defer os.Remove(binary)

	args := []string{
		"-tags=testtag",
		"-v", // We're going to look at the files it examines.
		"testdata/tagtest",
	}
	cmd = exec.Command("./"+binary, args...)
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
