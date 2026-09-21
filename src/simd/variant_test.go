// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package simd_test

import (
	"os"
	"os/exec"
	"runtime"
	"simd"
	"testing"
)

func TestVariant(t *testing.T) {
	if runtime.GOARCH != "arm64" {
		t.Skip("Variant test requires arm64")
		return
	}
	if simd.VectorBitSize() != 128 {
		t.Skip("Variant test requires 128-bit vectors")
		return
	}
	if !simd.HasHardwareCarrylessMultiply() {
		t.Skip("Variant test does not test itself")
		return
	}

	var args = []string{} // "-test.run=TestClMul"

	if testing.Verbose() {
		args = append(args, "-test.v")
	}

	cmd := exec.Command(os.Args[0], args...)

	cmd.Env = append(os.Environ(),
		// @variant undocumented testing option.
		"GODEBUG=simd=@nclm",
	)

	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("variant test failed: %v, output: %s", err, output)
	}

	t.Logf("Parent got output from child:\n%s", output)
}
