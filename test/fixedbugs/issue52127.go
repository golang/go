// run

//go:build !js && !wasip1

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 52127: Too many syntax errors in many files can
// cause deadlocks instead of displaying error messages
// correctly.

package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

func main() {
	dir, err := os.MkdirTemp("", "issue52127")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)

	args := []string{"go", "build"}
	write := func(prefix string, i int, data string) {
		filename := filepath.Join(dir, fmt.Sprintf("%s%d.go", prefix, i))
		if err := os.WriteFile(filename, []byte(data), 0o644); err != nil {
			panic(err)
		}
		args = append(args, filename)
	}

	for i := 0; i < 100; i++ {
		write("a", i, `package p
`)
	}
	for i := 0; i < 100; i++ {
		write("b", i, `package p
var
var
var
var
var
`)
	}

	cmd := exec.Command(args[0], args[1:]...)
	output, err := cmd.CombinedOutput()
	if err == nil {
		panic("compile succeeded unexpectedly")
	}
	if !bytes.Contains(output, []byte("syntax error:")) {
		panic(fmt.Sprintf(`missing "syntax error" in compiler output; got: 
%s`, output))
	}
}