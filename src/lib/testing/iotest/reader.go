// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The iotest package implements Readers and Writers
// useful only for testing.
package iotest

import (
	"io";
	"os";
)

type oneByteReader struct {
	r io.Reader;
}

func (r *oneByteReader) Read(p []byte) (int, os.Error) {
	if len(p) == 0 {
		return 0, nil;
	}
	return r.r.Read(p[0:1]);
}

// OneByteReader returns a Reader that implements
// each non-empty Read by reading one byte from r.
func OneByteReader(r io.Reader) io.Reader {
	return &oneByteReader{r};
}

type halfReader struct {
	r io.Reader;
}

func (r *halfReader) Read(p []byte) (int, os.Error) {
	return r.r.Read(p[0:(len(p)+1)/2]);
}

// HalfReader returns a Reader that implements Read
// by reading half as many requested bytes from r.
func HalfReader(r io.Reader) io.Reader {
	return &halfReader{r};
}

