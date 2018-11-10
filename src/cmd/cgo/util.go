// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"go/token"
	"io/ioutil"
	"os"
	"os/exec"
)

// run runs the command argv, feeding in stdin on standard input.
// It returns the output to standard output and standard error.
// ok indicates whether the command exited successfully.
func run(stdin []byte, argv []string) (stdout, stderr []byte, ok bool) {
	if i := find(argv, "-xc"); i >= 0 && argv[len(argv)-1] == "-" {
		// Some compilers have trouble with standard input.
		// Others have trouble with -xc.
		// Avoid both problems by writing a file with a .c extension.
		f, err := ioutil.TempFile("", "cgo-gcc-input-")
		if err != nil {
			fatalf("%s", err)
		}
		name := f.Name()
		f.Close()
		if err := ioutil.WriteFile(name+".c", stdin, 0666); err != nil {
			os.Remove(name)
			fatalf("%s", err)
		}
		defer os.Remove(name)
		defer os.Remove(name + ".c")

		// Build new argument list without -xc and trailing -.
		new := append(argv[:i:i], argv[i+1:len(argv)-1]...)

		// Since we are going to write the file to a temporary directory,
		// we will need to add -I . explicitly to the command line:
		// any #include "foo" before would have looked in the current
		// directory as the directory "holding" standard input, but now
		// the temporary directory holds the input.
		// We've also run into compilers that reject "-I." but allow "-I", ".",
		// so be sure to use two arguments.
		// This matters mainly for people invoking cgo -godefs by hand.
		new = append(new, "-I", ".")

		// Finish argument list with path to C file.
		new = append(new, name+".c")

		argv = new
		stdin = nil
	}

	p := exec.Command(argv[0], argv[1:]...)
	p.Stdin = bytes.NewReader(stdin)
	var bout, berr bytes.Buffer
	p.Stdout = &bout
	p.Stderr = &berr
	err := p.Run()
	if _, ok := err.(*exec.ExitError); err != nil && !ok {
		fatalf("%s", err)
	}
	ok = p.ProcessState.Success()
	stdout, stderr = bout.Bytes(), berr.Bytes()
	return
}

func find(argv []string, target string) int {
	for i, arg := range argv {
		if arg == target {
			return i
		}
	}
	return -1
}

func lineno(pos token.Pos) string {
	return fset.Position(pos).String()
}

// Die with an error message.
func fatalf(msg string, args ...interface{}) {
	// If we've already printed other errors, they might have
	// caused the fatal condition. Assume they're enough.
	if nerrors == 0 {
		fmt.Fprintf(os.Stderr, msg+"\n", args...)
	}
	os.Exit(2)
}

var nerrors int

func error_(pos token.Pos, msg string, args ...interface{}) {
	nerrors++
	if pos.IsValid() {
		fmt.Fprintf(os.Stderr, "%s: ", fset.Position(pos).String())
	}
	fmt.Fprintf(os.Stderr, msg, args...)
	fmt.Fprintf(os.Stderr, "\n")
}

// isName reports whether s is a valid C identifier
func isName(s string) bool {
	for i, v := range s {
		if v != '_' && (v < 'A' || v > 'Z') && (v < 'a' || v > 'z') && (v < '0' || v > '9') {
			return false
		}
		if i == 0 && '0' <= v && v <= '9' {
			return false
		}
	}
	return s != ""
}

func creat(name string) *os.File {
	f, err := os.Create(name)
	if err != nil {
		fatalf("%s", err)
	}
	return f
}

func slashToUnderscore(c rune) rune {
	if c == '/' || c == '\\' || c == ':' {
		c = '_'
	}
	return c
}
