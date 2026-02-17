// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package stdio_test

import (
	"bytes"
	"cmd/cgo/internal/cgotest"
	"internal/testenv"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestMain(m *testing.M) {
	log.SetFlags(log.Lshortfile)
	os.Exit(testMain(m))
}

func testMain(m *testing.M) int {
	GOPATH, err := os.MkdirTemp("", "cgostdio")
	if err != nil {
		log.Panic(err)
	}
	defer os.RemoveAll(GOPATH)
	os.Setenv("GOPATH", GOPATH)

	// Copy testdata into GOPATH/src/cgostdio, along with a go.mod file
	// declaring the same path.
	modRoot := filepath.Join(GOPATH, "src", "cgostdio")
	if err := cgotest.OverlayDir(modRoot, "testdata"); err != nil {
		log.Panic(err)
	}
	if err := os.Chdir(modRoot); err != nil {
		log.Panic(err)
	}
	os.Setenv("PWD", modRoot)
	if err := os.WriteFile("go.mod", []byte("module cgostdio\n"), 0666); err != nil {
		log.Panic(err)
	}

	return m.Run()
}

// TestTestRun runs a cgo test that doesn't depend on non-standard libraries.
func TestTestRun(t *testing.T) {
	testenv.MustHaveGoRun(t)
	testenv.MustHaveCGO(t)

	for _, file := range [...]string{
		"chain.go",
		"fib.go",
		"hello.go",
	} {
		file := file
		wantFile := strings.Replace(file, ".go", ".out", 1)
		t.Run(file, func(t *testing.T) {
			cmd := exec.Command(testenv.GoToolPath(t), "run", file)
			got, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("%v: %s\n%s", cmd, err, got)
			}
			got = bytes.ReplaceAll(got, []byte("\r\n"), []byte("\n"))
			want, err := os.ReadFile(wantFile)
			if err != nil {
				t.Fatal("reading golden output:", err)
			}
			if !bytes.Equal(got, want) {
				t.Errorf("'%v' output does not match expected in %s. Instead saw:\n%s", cmd, wantFile, got)
			}
		})
	}
}
