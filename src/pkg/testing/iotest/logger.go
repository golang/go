// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iotest

import (
	"io";
	"log";
	"os";
)

type writeLogger struct {
	prefix	string;
	w	io.Writer;
}

func (l *writeLogger) Write(p []byte) (n int, err os.Error) {
	n, err = l.w.Write(p);
	if err != nil {
		log.Stdoutf("%s %x: %v", l.prefix, p[0:n], err);
	} else {
		log.Stdoutf("%s %x", l.prefix, p[0:n]);
	}
	return;
}

// NewWriteLogger returns a writer that behaves like w except
// that it logs (using log.Stdout) each write to standard output,
// printing the prefix and the hexadecimal data written.
func NewWriteLogger(prefix string, w io.Writer) io.Writer {
	return &writeLogger{prefix, w};
}

type readLogger struct {
	prefix	string;
	r	io.Reader;
}

func (l *readLogger) Read(p []byte) (n int, err os.Error) {
	n, err = l.r.Read(p);
	if err != nil {
		log.Stdoutf("%s %x: %v", l.prefix, p[0:n], err);
	} else {
		log.Stdoutf("%s %x", l.prefix, p[0:n]);
	}
	return;
}

// NewReadLogger returns a writer that behaves like w except
// that it logs (using log.Stdout) each write to standard output,
// printing the prefix and the hexadecimal data written.
func NewReadLogger(prefix string, r io.Reader) io.Reader {
	return &readLogger{prefix, r};
}
