// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
)

const (
	dataDir = "testdata"
	binary  = "testvet.exe"
)

// We implement TestMain so remove the test binary when all is done.
func TestMain(m *testing.M) {
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
	buildMu sync.Mutex // guards following
	built   = false    // We have built the binary.
	failed  = false    // We have failed to build the binary, don't try again.
)

func Build(t *testing.T) {
	buildMu.Lock()
	defer buildMu.Unlock()
	if built {
		return
	}
	if failed {
		t.Skip("cannot run on this environment")
	}
	testenv.MustHaveGoBuild(t)
	MustHavePerl(t)
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", binary)
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

// TestVet tests self-contained files in testdata/*.go.
//
// If a file contains assembly or has inter-dependencies, it should be
// in its own test, like TestVetAsm, TestDivergentPackagesExamples,
// etc below.
func TestVet(t *testing.T) {
	Build(t)
	t.Parallel()

	// errchk ./testvet
	gos, err := filepath.Glob(filepath.Join(dataDir, "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	wide := runtime.GOMAXPROCS(0)
	if wide > len(gos) {
		wide = len(gos)
	}
	batch := make([][]string, wide)
	for i, file := range gos {
		batch[i%wide] = append(batch[i%wide], file)
	}
	for i, files := range batch {
		files := files
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			t.Parallel()
			t.Logf("files: %q", files)
			Vet(t, files)
		})
	}
}

func TestVetAsm(t *testing.T) {
	Build(t)

	asmDir := filepath.Join(dataDir, "asm")
	gos, err := filepath.Glob(filepath.Join(asmDir, "*.go"))
	if err != nil {
		t.Fatal(err)
	}
	asms, err := filepath.Glob(filepath.Join(asmDir, "*.s"))
	if err != nil {
		t.Fatal(err)
	}

	t.Parallel()
	// errchk ./testvet
	Vet(t, append(gos, asms...))
}

func TestVetDirs(t *testing.T) {
	t.Parallel()
	Build(t)
	for _, dir := range []string{
		"testingpkg",
		"divergent",
		"buildtag",
		"incomplete", // incomplete examples
		"cgo",
	} {
		dir := dir
		t.Run(dir, func(t *testing.T) {
			t.Parallel()
			gos, err := filepath.Glob(filepath.Join("testdata", dir, "*.go"))
			if err != nil {
				t.Fatal(err)
			}
			Vet(t, gos)
		})
	}
}

func run(c *exec.Cmd, t *testing.T) bool {
	output, err := c.CombinedOutput()
	if err != nil {
		t.Logf("vet output:\n%s", output)
		t.Fatal(err)
	}
	// Errchk delights by not returning non-zero status if it finds errors, so we look at the output.
	// It prints "BUG" if there is a failure.
	if !c.ProcessState.Success() {
		t.Logf("vet output:\n%s", output)
		return false
	}
	ok := !bytes.Contains(output, []byte("BUG"))
	if !ok {
		t.Logf("vet output:\n%s", output)
	}
	return ok
}

// TestTags verifies that the -tags argument controls which files to check.
func TestTags(t *testing.T) {
	t.Parallel()
	Build(t)
	for _, tag := range []string{"testtag", "x testtag y", "x,testtag,y"} {
		tag := tag
		t.Run(tag, func(t *testing.T) {
			t.Parallel()
			t.Logf("-tags=%s", tag)
			args := []string{
				"-tags=" + tag,
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
		})
	}
}

// Issue #21188.
func TestVetVerbose(t *testing.T) {
	t.Parallel()
	Build(t)
	cmd := exec.Command("./"+binary, "-v", "-all", "testdata/cgo/cgo3.go")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Logf("%s", out)
		t.Error(err)
	}
}
