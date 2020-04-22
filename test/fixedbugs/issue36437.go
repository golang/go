// run

// +build !nacl,!js

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests that when non-existent files are passed to the
// compiler, such as in:
//    go tool compile foo
// we don't print the beginning position:
//    foo:0: open foo: no such file or directory
// but instead omit it and print out:
//    open foo: no such file or directory

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

func main() {
	tmpDir, err := ioutil.TempDir("", "issue36437")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(tmpDir)

	msgOrErr := func(msg []byte, err error) string {
		if len(msg) == 0 && err != nil {
			return err.Error()
		}
		return string(msg)
	}

	// 1. Pass in a non-existent file.
	output, err := exec.Command("go", "tool", "compile", "x.go").CombinedOutput()
	want := "open x.go: no such file or directory\n"
	if got := msgOrErr(output, err); got != want {
		fmt.Printf("Expected an error, but got:\n\t%q\nwant:\n\t%q", got, want)
		return
	}

	if runtime.GOOS == "linux" && runtime.GOARCH == "amd64" {
		// The Go Linux builders seem to be running under root, thus
		// linux-amd64 doesn't seem to be respecting 0222 file permissions,
		// and reads files with -W-*-W-*-W- permissions.
		// Filed bug: https://golang.org/issues/38608
		return
	}

	// 2. Invoke the compiler with a file that we don't have read permissions to.
	path := filepath.Join(tmpDir, "p.go")
	if err := ioutil.WriteFile(path, []byte("package p"), 0222); err != nil {
		panic(err)
	}
	output, err = exec.Command("go", "tool", "compile", path).CombinedOutput()
	want = fmt.Sprintf("open %s: permission denied\n", path)
	if got := msgOrErr(output, err); got != want {
		fmt.Printf("Expected an error, but got:\n\t%q\nwant:\n\t%q", got, want)
		return
	}
}
