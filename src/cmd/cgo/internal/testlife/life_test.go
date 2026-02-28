// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package life_test

import (
	"bytes"
	"cmd/cgo/internal/cgotest"
	"internal/testenv"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestMain(m *testing.M) {
	log.SetFlags(log.Lshortfile)
	os.Exit(testMain(m))
}

func testMain(m *testing.M) int {
	GOPATH, err := os.MkdirTemp("", "cgolife")
	if err != nil {
		log.Panic(err)
	}
	defer os.RemoveAll(GOPATH)
	os.Setenv("GOPATH", GOPATH)

	// Copy testdata into GOPATH/src/cgolife, along with a go.mod file
	// declaring the same path.
	modRoot := filepath.Join(GOPATH, "src", "cgolife")
	if err := cgotest.OverlayDir(modRoot, "testdata"); err != nil {
		log.Panic(err)
	}
	if err := os.Chdir(modRoot); err != nil {
		log.Panic(err)
	}
	os.Setenv("PWD", modRoot)
	if err := os.WriteFile("go.mod", []byte("module cgolife\n"), 0666); err != nil {
		log.Panic(err)
	}

	return m.Run()
}

// TestTestRun runs a test case for cgo //export.
func TestTestRun(t *testing.T) {
	testenv.MustHaveGoRun(t)
	testenv.MustHaveCGO(t)

	cmd := exec.Command(testenv.GoToolPath(t), "run", "main.go")
	got, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%v: %s\n%s", cmd, err, got)
	}
	want, err := os.ReadFile("main.out")
	if err != nil {
		t.Fatal("reading golden output:", err)
	}
	if !bytes.Equal(got, want) {
		t.Errorf("'%v' output does not match expected in main.out. Instead saw:\n%s", cmd, got)
	}
}
