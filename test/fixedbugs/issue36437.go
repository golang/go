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
	"regexp"
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

	filename := "non-existent.go"
	output, err := exec.Command("go", "tool", "compile", filename).CombinedOutput()
	got := msgOrErr(output, err)

	regFilenamePos := regexp.MustCompile(filename + ":\\d+")
	if regFilenamePos.MatchString(got) {
		fmt.Printf("Error message must not contain filename:pos, but got:\n%q\n", got)
	}
}
