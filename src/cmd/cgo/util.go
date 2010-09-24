// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"exec"
	"fmt"
	"go/token"
	"io/ioutil"
	"os"
)

// run runs the command argv, feeding in stdin on standard input.
// It returns the output to standard output and standard error.
// ok indicates whether the command exited successfully.
func run(stdin []byte, argv []string) (stdout, stderr []byte, ok bool) {
	cmd, err := exec.LookPath(argv[0])
	if err != nil {
		fatal("exec %s: %s", argv[0], err)
	}
	r0, w0, err := os.Pipe()
	if err != nil {
		fatal("%s", err)
	}
	r1, w1, err := os.Pipe()
	if err != nil {
		fatal("%s", err)
	}
	r2, w2, err := os.Pipe()
	if err != nil {
		fatal("%s", err)
	}
	pid, err := os.ForkExec(cmd, argv, os.Environ(), "", []*os.File{r0, w1, w2})
	if err != nil {
		fatal("%s", err)
	}
	r0.Close()
	w1.Close()
	w2.Close()
	c := make(chan bool)
	go func() {
		w0.Write(stdin)
		w0.Close()
		c <- true
	}()
	var xstdout []byte // TODO(rsc): delete after 6g can take address of out parameter
	go func() {
		xstdout, _ = ioutil.ReadAll(r1)
		r1.Close()
		c <- true
	}()
	stderr, _ = ioutil.ReadAll(r2)
	r2.Close()
	<-c
	<-c
	stdout = xstdout

	w, err := os.Wait(pid, 0)
	if err != nil {
		fatal("%s", err)
	}
	ok = w.Exited() && w.ExitStatus() == 0
	return
}

// Die with an error message.
func fatal(msg string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, msg+"\n", args...)
	os.Exit(2)
}

var nerrors int
var noPos token.Position

func error(pos token.Position, msg string, args ...interface{}) {
	nerrors++
	if pos.IsValid() {
		fmt.Fprintf(os.Stderr, "%s: ", pos)
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
	f, err := os.Open(name, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0666)
	if err != nil {
		fatal("%s", err)
	}
	return f
}

func slashToUnderscore(c int) int {
	if c == '/' {
		c = '_'
	}
	return c
}

func concat(a, b []string) []string {
	c := make([]string, len(a)+len(b))
	copy(c, a)
	copy(c[len(a):], b)
	return c
}
