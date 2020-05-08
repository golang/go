// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big

import (
	"bytes"
	"internal/testenv"
	"io/ioutil"
	"os/exec"
	"path/filepath"
	"testing"
)

// Tests that the linker is able to remove references to Float, Rat,
// and Int if unused (notably, not used by init).
func TestLinkerGC(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	t.Parallel()
	goBin := testenv.GoToolPath(t)
	goFile := filepath.Join(t.TempDir(), "x.go")
	file := []byte(`package main
import _ "math/big"
func main() {}
`)
	if err := ioutil.WriteFile(goFile, file, 0644); err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(goBin, "build", "-o", "x.exe", "x.go")
	cmd.Dir = t.TempDir()
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile: %v, %s", err, out)
	}

	cmd = exec.Command(goBin, "tool", "nm", "x.exe")
	cmd.Dir = t.TempDir()
	nm, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("nm: %v, %s", err, nm)
	}
	const want = "runtime.(*Frames).Next"
	if !bytes.Contains(nm, []byte(want)) {
		// Test the test.
		t.Errorf("expected symbol %q not found", want)
	}
	bad := []string{
		"math/big.(*Float)",
		"math/big.(*Rat)",
		"math/big.(*Int)",
	}
	for _, sym := range bad {
		if bytes.Contains(nm, []byte(sym)) {
			t.Errorf("unexpected symbol %q found", sym)
		}
	}
	if t.Failed() {
		t.Logf("Got: %s", nm)
	}
}
