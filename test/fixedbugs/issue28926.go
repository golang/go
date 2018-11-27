// +build !nacl,!js
// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// ensure that panic(x) where x is a numeric type displays a readable number
package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
)

const fn = `
package main

func main() {
	var e interface{}
	switch e := e.(type) {
	case G:
		e.M()
	case E:
		e.D()
	}
}
`

func main() {
	tempDir, err := ioutil.TempDir("", "")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(tempDir)
	tmpFile := filepath.Join(tempDir, "tmp28926.go")

	b := bytes.Buffer{}
	fmt.Fprintf(&b, fn)

	err = ioutil.WriteFile(tmpFile, b.Bytes(), 0644)
	if err != nil {
		log.Fatal(err)
	}

	cmd := exec.Command("go", "tool", "compile", tmpFile)
	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf
	cmd.Env = os.Environ()
	cmd.Run() // ignore err as we expect a panic

	out := buf.Bytes()
	firstTypeCheck := bytes.Index(out, []byte("undefined: G"))
	if firstTypeCheck == -1 {
		log.Fatalf("expected undefined type check error for G in output. got: %s", out)
	}
	secondTypeCheck := bytes.Index(out, []byte("undefined: E"))
	if secondTypeCheck == -1 {
		log.Fatalf("expected second undefined for E, got: %s", out)
	}
	inCaseTypeError := bytes.Index(out, []byte("e.M undefined (type interface {} is interface with no methods)"))
	if inCaseTypeError != -1 {
		log.Fatalf("expected error to not to appear in output, got: %s", out)
	}
}
