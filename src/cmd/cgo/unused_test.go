// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"os/exec"
	"testing"
)

func TestUnusedParameter(t *testing.T) {

	filename := "tmp.go"
	file, err := os.Create(filename)
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(filename)
	defer file.Close()

	code := `
		package p

		// #cgo CFLAGS: -Werror=unused-parameter
		import "C"
	`
	file.WriteString(code)

	cmd := exec.Command("../../../bin/go", "build", filename)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 0 {
		t.Fatalf("unexpected output: %s", string(out))
	}

}
