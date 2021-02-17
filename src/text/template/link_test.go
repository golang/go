// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template_test

import (
	"bytes"
	"internal/testenv"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// Issue 36021: verify that text/template doesn't prevent the linker from removing
// unused methods.
func TestLinkerGC(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	testenv.MustHaveGoBuild(t)
	const prog = `package main

import (
	_ "text/template"
)

type T struct{}

func (t *T) Unused() { println("THIS SHOULD BE ELIMINATED") }
func (t *T) Used() {}

var sink *T

func main() {
	var t T
	sink = &t
	t.Used()
}
`
	td, err := os.MkdirTemp("", "text_template_TestDeadCodeElimination")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(td)

	if err := os.WriteFile(filepath.Join(td, "x.go"), []byte(prog), 0644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(testenv.GoToolPath(t), "build", "-o", "x.exe", "x.go")
	cmd.Dir = td
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("go build: %v, %s", err, out)
	}
	slurp, err := os.ReadFile(filepath.Join(td, "x.exe"))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(slurp, []byte("THIS SHOULD BE ELIMINATED")) {
		t.Error("binary contains code that should be deadcode eliminated")
	}
}
