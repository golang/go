// run

//go:build !js && !wasip1

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const src = `package p

//go:noinline
func sink(i int, f float64) {}

func div(i int, x, y float64) {
	sink(i, x/y)
}
`

func isSoftFloat(s string) bool {
	for _, v := range strings.Split(s, ",") {
		if v == "softfloat" {
			return true
		}
	}
	return false
}

func main() {
	for _, env := range []string{"GO386", "GOMIPS", "GOMIPS64", "GOARM"} {
		if isSoftFloat(os.Getenv(env)) {
			return
		}
	}

	dir, err := os.MkdirTemp("", "issue28698")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)

	fn := filepath.Join(dir, "p.go")
	if err := os.WriteFile(fn, []byte(src), 0644); err != nil {
		panic(err)
	}

	// Float division does not require a temporary when preparing call arguments.
	cmd := exec.Command("go", "tool", "compile", "-W", fn)
	out, err := cmd.CombinedOutput()
	if err != nil {
		panic(err)
	}

	if bytes.Contains(out, []byte(".autotmp_")) {
		fmt.Println(string(out))
		panic("unexpected autotmp")
	}

	// Forcing softfloat should still emit temporary.
	cmd = exec.Command("go", "tool", "compile", "-W", "-d=softfloat", fn)
	out, err = cmd.CombinedOutput()
	if err != nil {
		panic(err)
	}

	if !bytes.Contains(out, []byte(".autotmp_")) {
		fmt.Println(string(out))
		panic("no autotmp")
	}
}
