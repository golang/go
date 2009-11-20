// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"exec";
	"fmt";
	"go/token";
	"io";
	"os";
)

// A ByteReaderAt implements io.ReadAt using a slice of bytes.
type ByteReaderAt []byte

func (r ByteReaderAt) ReadAt(p []byte, off int64) (n int, err os.Error) {
	if off >= int64(len(r)) || off < 0 {
		return 0, os.EOF
	}
	return copy(p, r[off:]), nil;
}

// run runs the command argv, feeding in stdin on standard input.
// It returns the output to standard output and standard error.
// ok indicates whether the command exited successfully.
func run(stdin []byte, argv []string) (stdout, stderr []byte, ok bool) {
	cmd, err := exec.LookPath(argv[0]);
	if err != nil {
		fatal("exec %s: %s", argv[0], err)
	}
	r0, w0, err := os.Pipe();
	if err != nil {
		fatal("%s", err)
	}
	r1, w1, err := os.Pipe();
	if err != nil {
		fatal("%s", err)
	}
	r2, w2, err := os.Pipe();
	if err != nil {
		fatal("%s", err)
	}
	pid, err := os.ForkExec(cmd, argv, os.Environ(), "", []*os.File{r0, w1, w2});
	if err != nil {
		fatal("%s", err)
	}
	r0.Close();
	w1.Close();
	w2.Close();
	c := make(chan bool);
	go func() {
		w0.Write(stdin);
		w0.Close();
		c <- true;
	}();
	var xstdout []byte;	// TODO(rsc): delete after 6g can take address of out parameter
	go func() {
		xstdout, _ = io.ReadAll(r1);
		r1.Close();
		c <- true;
	}();
	stderr, _ = io.ReadAll(r2);
	r2.Close();
	<-c;
	<-c;
	stdout = xstdout;

	w, err := os.Wait(pid, 0);
	if err != nil {
		fatal("%s", err)
	}
	ok = w.Exited() && w.ExitStatus() == 0;
	return;
}

// Die with an error message.
func fatal(msg string, args ...) {
	fmt.Fprintf(os.Stderr, msg+"\n", args);
	os.Exit(2);
}

var nerrors int
var noPos token.Position

func error(pos token.Position, msg string, args ...) {
	nerrors++;
	if pos.IsValid() {
		fmt.Fprintf(os.Stderr, "%s: ", pos)
	}
	fmt.Fprintf(os.Stderr, msg, args);
	fmt.Fprintf(os.Stderr, "\n");
}
