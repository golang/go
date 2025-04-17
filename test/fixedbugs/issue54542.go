// run

//go:build !js && !wasip1

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

const aSrc = `package a

func A() { println("a") }
`

const mainSrc = `package main

import "a"

func main() { a.A() }
`

var srcs = map[string]string{
	"a.go":    aSrc,
	"main.go": mainSrc,
}

func main() {
	dir, err := os.MkdirTemp("", "issue54542")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)

	for fn, src := range srcs {
		if err := os.WriteFile(filepath.Join(dir, fn), []byte(src), 0644); err != nil {
			panic(err)
		}
	}

	if _, err := runInDir(dir, "tool", "compile", "-p=lie", "a.go"); err != nil {
		panic(err)
	}

	out, err := runInDir(dir, "tool", "compile", "-I=.", "-p=main", "main.go")
	if err == nil {
		panic("compiling succeed unexpectedly")
	}

	if bytes.Contains(out, []byte("internal compiler error:")) {
		panic(fmt.Sprintf("unexpected ICE:\n%s", string(out)))
	}
}

func runInDir(dir string, args ...string) ([]byte, error) {
	cmd := exec.Command("go", args...)
	cmd.Dir = dir
	return cmd.CombinedOutput()
}
