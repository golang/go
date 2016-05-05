// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package iotest implements Readers and Writers useful mainly for testing.
package iotest

import (
	"errors"
	"io"
)

// OneByteReader returns a Reader that implements
// each non-empty Read by reading one byte from r.
func OneByteReader(r io.Reader) io.Reader { return &oneByteReader{r} }

type oneByteReader struct {
	r io.Reader
}

func (r *oneByteReader) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	return r.r.Read(p[0:1])
}

// HalfReader returns a Reader that implements Read
// by reading half as many requested bytes from r.
func HalfReader(r io.Reader) io.Reader { return &halfReader{r} }

type halfReader struct {
	r io.Reader
}

func (r *halfReader) Read(p []byte) (int, error) {
	return r.r.Read(p[0 : (len(p)+1)/2])
}

// DataErrReader changes the way errors are handled by a Reader. Normally, a
// Reader returns an error (typically EOF) from the first Read call after the
// last piece of data is read. DataErrReader wraps a Reader and changes its
// behavior so the final error is returned along with the final data, instead
// of in the first call after the final data.
func DataErrReader(r io.Reader) io.Reader { return &dataErrReader{r, nil, make([]byte, 1024)} }

type dataErrReader struct {
	r      io.Reader
	unread []byte
	data   []byte
}

func (r *dataErrReader) Read(p []byte) (n int, err error) {
	// loop because first call needs two reads:
	// one to get data and a second to look for an error.
	for {
		if len(r.unread) == 0 {
			n1, err1 := r.r.Read(r.data)
			r.unread = r.data[0:n1]
			err = err1
		}
		if n > 0 || err != nil {
			break
		}
		n = copy(p, r.unread)
		r.unread = r.unread[n:]
	}
	return
}

var ErrTimeout = errors.New("timeout")

// TimeoutReader returns ErrTimeout on the second read
// with no data. Subsequent calls to read succeed.
func TimeoutReader(r io.Reader) io.Reader { return &timeoutReader{r, 0} }

type timeoutReader struct {
	r     io.Reader
	count int
}

func (r *timeoutReader) Read(p []byte) (int, error) {
	r.count++
	if r.count == 2 {
		return 0, ErrTimeout
	}
	return r.r.Read(p)
}
