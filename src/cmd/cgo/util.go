// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"go/token"
	"os"
	"os/exec"
)

// run runs the command argv, feeding in stdin on standard input.
// It returns the output to standard output and standard error.
// ok indicates whether the command exited successfully.
func run(stdin []byte, argv []string) (stdout, stderr []byte, ok bool) {
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

func lineno(pos token.Pos) string {
	return fset.Position(pos).String()
}

// Die with an error message.
func fatalf(msg string, args ...interface{}) {
	// If we've already printed other errors, they might have
	// caused the fatal condition.  Assume they're enough.
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

// isName returns true if s is a valid C identifier
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
