// run
//go:build !nacl && !js && !wasip1 && !gccgo

// Copyright 2017 The Go Authors. All rights reserved.
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

import  "errors"
type S struct {

}
func (s S) String() string {
	return "s-stringer"
}
func main() {
 	_ = errors.New
  panic(%s(%s))
}
`

func main() {
	tempDir, err := ioutil.TempDir("", "")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(tempDir)
	tmpFile := filepath.Join(tempDir, "tmp.go")

	for _, tc := range []struct {
		Type   string
		Input  string
		Expect string
	}{
		{"", "nil", "panic: runtime error: panic called with nil argument"},
		{"errors.New", `"test"`, "panic: test"},
		{"S", "S{}", "panic: s-stringer"},
		{"byte", "8", "panic: 8"},
		{"rune", "8", "panic: 8"},
		{"int", "8", "panic: 8"},
		{"int8", "8", "panic: 8"},
		{"int16", "8", "panic: 8"},
		{"int32", "8", "panic: 8"},
		{"int64", "8", "panic: 8"},
		{"uint", "8", "panic: 8"},
		{"uint8", "8", "panic: 8"},
		{"uint16", "8", "panic: 8"},
		{"uint32", "8", "panic: 8"},
		{"uint64", "8", "panic: 8"},
		{"uintptr", "8", "panic: 8"},
		{"bool", "true", "panic: true"},
		{"complex64", "8 + 16i", "panic: (+8.000000e+000+1.600000e+001i)"},
		{"complex128", "8+16i", "panic: (+8.000000e+000+1.600000e+001i)"},
		{"string", `"test"`, "panic: test"}} {

		b := bytes.Buffer{}
		fmt.Fprintf(&b, fn, tc.Type, tc.Input)

		err = ioutil.WriteFile(tmpFile, b.Bytes(), 0644)
		if err != nil {
			log.Fatal(err)
		}

		cmd := exec.Command("go", "run", tmpFile)
		var buf bytes.Buffer
		cmd.Stdout = &buf
		cmd.Stderr = &buf
		cmd.Env = os.Environ()
		cmd.Run() // ignore err as we expect a panic

		out := buf.Bytes()
		panicIdx := bytes.Index(out, []byte("panic: "))
		if panicIdx == -1 {
			log.Fatalf("expected a panic in output for %s, got: %s", tc.Type, out)
		}
		eolIdx := bytes.IndexByte(out[panicIdx:], '\n') + panicIdx
		if panicIdx == -1 {
			log.Fatalf("expected a newline in output for %s after the panic, got: %s", tc.Type, out)
		}
		out = out[0:eolIdx]
		if string(out) != tc.Expect {
			log.Fatalf("expected '%s' for panic(%s(%s)), got %s", tc.Expect, tc.Type, tc.Input, out)
		}
	}
}
