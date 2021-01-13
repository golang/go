// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import (
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func test43639(t *testing.T) {
	dir, err := ioutil.TempDir("", filepath.Base(t.Name()))
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)

	filename := filepath.Join(dir, "tmp.go")
	file, err := os.Create(filename)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()

	code := `
		package p

		// #cgo CFLAGS: -W -Wall -Werror
		import "C"
	`
	file.WriteString(code)

	cmd := exec.Command("go", "build", filename)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 0 {
		t.Fatalf("unexpected output: %s", string(out))
	}
}
