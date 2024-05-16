// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loong64

import (
	"bytes"
	"fmt"
	"internal/testenv"
	"os"
	"path/filepath"
	"testing"
)

const genBufSize = (1024 * 1024 * 32) // 32MB

// TestLargeBranch generates a large function with a very far conditional
// branch, in order to ensure that it assembles successfully.
func TestLargeBranch(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test in short mode")
	}
	testenv.MustHaveGoBuild(t)

	dir, err := os.MkdirTemp("", "testlargebranch")
	if err != nil {
		t.Fatalf("Could not create directory: %v", err)
	}
	defer os.RemoveAll(dir)

	// Generate a very large function.
	buf := bytes.NewBuffer(make([]byte, 0, genBufSize))
	genLargeBranch(buf)

	tmpfile := filepath.Join(dir, "x.s")
	if err := os.WriteFile(tmpfile, buf.Bytes(), 0644); err != nil {
		t.Fatalf("Failed to write file: %v", err)
	}

	// Assemble generated file.
	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "asm", "-o", filepath.Join(dir, "x.o"), tmpfile)
	cmd.Env = append(os.Environ(), "GOARCH=loong64", "GOOS=linux")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Errorf("Build failed: %v, output: %s", err, out)
	}
}

func genLargeBranch(buf *bytes.Buffer) {
	genSize1 := (1 << 16) + 16
	genSize2 := (1 << 21) + 16

	fmt.Fprintln(buf, "TEXT f(SB),0,$0-0")
	fmt.Fprintln(buf, "BEQ R5, R6, label18")
	fmt.Fprintln(buf, "BNE R5, R6, label18")
	fmt.Fprintln(buf, "BGE R5, R6, label18")

	fmt.Fprintln(buf, "BGEU R5, R6, label18")
	fmt.Fprintln(buf, "BLTU R5, R6, label18")

	fmt.Fprintln(buf, "BLEZ R5, label18")
	fmt.Fprintln(buf, "BGEZ R5, label18")
	fmt.Fprintln(buf, "BLTZ R5, label18")
	fmt.Fprintln(buf, "BGTZ R5, label18")

	fmt.Fprintln(buf, "BFPT label23")
	fmt.Fprintln(buf, "BFPF label23")

	fmt.Fprintln(buf, "BEQ R5, label23")
	fmt.Fprintln(buf, "BNE R5, label23")

	for i := 0; i <= genSize1; i++ {
		fmt.Fprintln(buf, "ADDV $0, R0, R0")
	}

	fmt.Fprintln(buf, "label18:")
	for i := 0; i <= (genSize2 - genSize1); i++ {
		fmt.Fprintln(buf, "ADDV $0, R0, R0")
	}

	fmt.Fprintln(buf, "label23:")
	fmt.Fprintln(buf, "ADDV $0, R0, R0")
	fmt.Fprintln(buf, "RET")
}
