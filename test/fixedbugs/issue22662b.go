// +build !js,gc
// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify the impact of line directives on error positions and position formatting.

package main

import (
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"strings"
)

// Each of these tests is expected to fail (missing package clause)
// at the position determined by the preceding line directive.
var tests = []struct {
	src, pos string
}{
	{"//line :10\n", ":10:"},                   // no filename means no filename
	{"//line :10:4\n", "filename:10:4"},        // no filename means use existing filename
	{"//line foo.go:10\n", "foo.go:10:"},       // no column means don't print a column
	{"//line foo.go:10:4\n", "foo.go:10:4:"},   // column means print a column
	{"//line foo.go:10:4\n\n", "foo.go:11:1:"}, // relative columns start at 1 after newline

	{"/*line :10*/", ":10:"},
	{"/*line :10:4*/", "filename:10:4"},
	{"/*line foo.go:10*/", "foo.go:10:"},
	{"/*line foo.go:10:4*/", "foo.go:10:4:"},
	{"/*line foo.go:10:4*/\n", "foo.go:11:1:"},
}

func main() {
	f, err := ioutil.TempFile("", "issue22662b.go")
	if err != nil {
		log.Fatal(err)
	}
	f.Close()
	defer os.Remove(f.Name())

	for _, test := range tests {
		if err := ioutil.WriteFile(f.Name(), []byte(test.src), 0660); err != nil {
			log.Fatal(err)
		}

		out, err := exec.Command("go", "tool", "compile", f.Name()).CombinedOutput()
		if err == nil {
			log.Fatalf("expected compiling\n---\n%s\n---\nto fail", test.src)
		}

		errmsg := strings.Replace(string(out), f.Name(), "filename", -1) // use "filename" instead of actual (long) filename
		if !strings.HasPrefix(errmsg, test.pos) {
			log.Fatalf("%q: got %q; want position %q", test.src, errmsg, test.pos)
		}
	}
}
