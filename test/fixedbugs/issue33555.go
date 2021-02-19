// +build !nacl,!js,!gccgo
// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that the linker permits long call sequences.
package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
)

const start = `
package main

func main() {
	println(f0() + 1)
}
`

const fn = `
//go:noinline
func f%d() int {
	return f%d() + 1
}`

const fnlast = `
//go:noinline
func f%d() int {
	return 0
}
`

const count = 400

func main() {
	if err := test(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func test() error {
	var buf bytes.Buffer
	buf.WriteString(start)
	for i := 0; i < count; i++ {
		fmt.Fprintf(&buf, fn, i, i + 1)
	}
	fmt.Fprintf(&buf, fnlast, count)

	dir, err := ioutil.TempDir("", "issue33555")
	if err != nil {
		return err
	}
	defer os.RemoveAll(dir)

	fn := filepath.Join(dir, "x.go")
	if err := ioutil.WriteFile(fn, buf.Bytes(), 0644); err != nil {
		return err
	}

	out, err := exec.Command("go", "run", fn).CombinedOutput()
	if err != nil {
		return err
	}

	want := strconv.Itoa(count + 1)
	if got := string(bytes.TrimSpace(out)); got != want {
		return fmt.Errorf("got %q want %q", got, want)
	}

	return nil
}
