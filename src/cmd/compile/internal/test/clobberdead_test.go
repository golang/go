// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"internal/testenv"
	"io/ioutil"
	"os/exec"
	"path/filepath"
	"testing"
)

const helloSrc = `
package main
import "fmt"
func main() { fmt.Println("hello") }
`

func TestClobberDead(t *testing.T) {
	// Test that clobberdead mode generates correct program.
	runHello(t, "-clobberdead")
}

func TestClobberDeadReg(t *testing.T) {
	// Test that clobberdeadreg mode generates correct program.
	runHello(t, "-clobberdeadreg")
}

func runHello(t *testing.T, flag string) {
	if testing.Short() {
		// This test rebuilds the runtime with a special flag, which
		// takes a while.
		t.Skip("skip in short mode")
	}
	testenv.MustHaveGoRun(t)
	t.Parallel()

	tmpdir := t.TempDir()
	src := filepath.Join(tmpdir, "x.go")
	err := ioutil.WriteFile(src, []byte(helloSrc), 0644)
	if err != nil {
		t.Fatalf("write file failed: %v", err)
	}

	cmd := exec.Command(testenv.GoToolPath(t), "run", "-gcflags=all="+flag, src)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("go run failed: %v\n%s", err, out)
	}
	if string(out) != "hello\n" {
		t.Errorf("wrong output: got %q, want %q", out, "hello\n")
	}
}
