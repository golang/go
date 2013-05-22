// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main_test

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
)

const (
	dataDir = "testdata"
	binary  = "testvet"
)

// Run this shell script, but do it in Go so it can be run by "go test".
// 	go build -o testvet
// 	$(GOROOT)/test/errchk ./testvet -printfuncs='Warn:1,Warnf:1' testdata/*.go testdata/*.s
// 	rm testvet
//
func TestVet(t *testing.T) {
	// Windows systems can't be guaranteed to have Perl and so can't run errchk.
	if runtime.GOOS == "windows" {
		t.Skip("skipping test; no Perl on Windows")
	}

	// go build
	cmd := exec.Command("go", "build", "-o", binary)
	run(cmd, t)

	// defer removal of vet
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
	errchk := filepath.Join(runtime.GOROOT(), "test", "errchk")
	flags := []string{
		binary,
		"-printfuncs=Warn:1,Warnf:1",
	}
	cmd = exec.Command(errchk, append(flags, files...)...)
	run(cmd, t)
}

func run(c *exec.Cmd, t *testing.T) {
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr
	err := c.Run()
	if err != nil {
		t.Fatal(err)
	}
}
