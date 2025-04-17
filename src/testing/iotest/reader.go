// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package iotest implements Readers and Writers useful mainly for testing.
package iotest

import (
	"bytes"
	"errors"
	"fmt"
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

// ErrTimeout is a fake timeout error.
var ErrTimeout = errors.New("timeout")

// TimeoutReader returns [ErrTimeout] on the second read
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

// ErrReader returns an [io.Reader] that returns 0, err from all Read calls.
func ErrReader(err error) io.Reader {
	return &errReader{err: err}
}

type errReader struct {
	err error
}

func (r *errReader) Read(p []byte) (int, error) {
	return 0, r.err
}

type smallByteReader struct {
	r   io.Reader
	off int
	n   int
}

func (r *smallByteReader) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	r.n = r.n%3 + 1
	n := r.n
	if n > len(p) {
		n = len(p)
	}
	n, err := r.r.Read(p[0:n])
	if err != nil && err != io.EOF {
		err = fmt.Errorf("Read(%d bytes at offset %d): %v", n, r.off, err)
	}
	r.off += n
	return n, err
}

// TestReader tests that reading from r returns the expected file content.
// It does reads of different sizes, until EOF.
// If r implements [io.ReaderAt] or [io.Seeker], TestReader also checks
// that those operations behave as they should.
//
// If TestReader finds any misbehaviors, it returns an error reporting them.
// The error text may span multiple lines.
func TestReader(r io.Reader, content []byte) error {
	if len(content) > 0 {
		n, err := r.Read(nil)
		if n != 0 || err != nil {
			return fmt.Errorf("Read(0) = %d, %v, want 0, nil", n, err)
		}
	}

	data, err := io.ReadAll(&smallByteReader{r: r})
	if err != nil {
		return err
	}
	if !bytes.Equal(data, content) {
		return fmt.Errorf("ReadAll(small amounts) = %q\n\twant %q", data, content)
	}
	n, err := r.Read(make([]byte, 10))
	if n != 0 || err != io.EOF {
		return fmt.Errorf("Read(10) at EOF = %v, %v, want 0, EOF", n, err)
	}

	if r, ok := r.(io.ReadSeeker); ok {
		// Seek(0, 1) should report the current file position (EOF).
		if off, err := r.Seek(0, 1); off != int64(len(content)) || err != nil {
			return fmt.Errorf("Seek(0, 1) from EOF = %d, %v, want %d, nil", off, err, len(content))
		}

		// Seek backward partway through file, in two steps.
		// If middle == 0, len(content) == 0, can't use the -1 and +1 seeks.
		middle := len(content) - len(content)/3
		if middle > 0 {
			if off, err := r.Seek(-1, 1); off != int64(len(content)-1) || err != nil {
				return fmt.Errorf("Seek(-1, 1) from EOF = %d, %v, want %d, nil", -off, err, len(content)-1)
			}
			if off, err := r.Seek(int64(-len(content)/3), 1); off != int64(middle-1) || err != nil {
				return fmt.Errorf("Seek(%d, 1) from %d = %d, %v, want %d, nil", -len(content)/3, len(content)-1, off, err, middle-1)
			}
			if off, err := r.Seek(+1, 1); off != int64(middle) || err != nil {
				return fmt.Errorf("Seek(+1, 1) from %d = %d, %v, want %d, nil", middle-1, off, err, middle)
			}
		}

		// Seek(0, 1) should report the current file position (middle).
		if off, err := r.Seek(0, 1); off != int64(middle) || err != nil {
			return fmt.Errorf("Seek(0, 1) from %d = %d, %v, want %d, nil", middle, off, err, middle)
		}

		// Reading forward should return the last part of the file.
		data, err := io.ReadAll(&smallByteReader{r: r})
		if err != nil {
			return fmt.Errorf("ReadAll from offset %d: %v", middle, err)
		}
		if !bytes.Equal(data, content[middle:]) {
			return fmt.Errorf("ReadAll from offset %d = %q\n\twant %q", middle, data, content[middle:])
		}

		// Seek relative to end of file, but start elsewhere.
		if off, err := r.Seek(int64(middle/2), 0); off != int64(middle/2) || err != nil {
			return fmt.Errorf("Seek(%d, 0) from EOF = %d, %v, want %d, nil", middle/2, off, err, middle/2)
		}
		if off, err := r.Seek(int64(-len(content)/3), 2); off != int64(middle) || err != nil {
			return fmt.Errorf("Seek(%d, 2) from %d = %d, %v, want %d, nil", -len(content)/3, middle/2, off, err, middle)
		}

		// Reading forward should return the last part of the file (again).
		data, err = io.ReadAll(&smallByteReader{r: r})
		if err != nil {
			return fmt.Errorf("ReadAll from offset %d: %v", middle, err)
		}
		if !bytes.Equal(data, content[middle:]) {
			return fmt.Errorf("ReadAll from offset %d = %q\n\twant %q", middle, data, content[middle:])
		}

		// Absolute seek & read forward.
		if off, err := r.Seek(int64(middle/2), 0); off != int64(middle/2) || err != nil {
			return fmt.Errorf("Seek(%d, 0) from EOF = %d, %v, want %d, nil", middle/2, off, err, middle/2)
		}
		data, err = io.ReadAll(r)
		if err != nil {
			return fmt.Errorf("ReadAll from offset %d: %v", middle/2, err)
		}
		if !bytes.Equal(data, content[middle/2:]) {
			return fmt.Errorf("ReadAll from offset %d = %q\n\twant %q", middle/2, data, content[middle/2:])
		}
	}

	if r, ok := r.(io.ReaderAt); ok {
		data := make([]byte, len(content), len(content)+1)
		for i := range data {
			data[i] = 0xfe
		}
		n, err := r.ReadAt(data, 0)
		if n != len(data) || err != nil && err != io.EOF {
			return fmt.Errorf("ReadAt(%d, 0) = %v, %v, want %d, nil or EOF", len(data), n, err, len(data))
		}
		if !bytes.Equal(data, content) {
			return fmt.Errorf("ReadAt(%d, 0) = %q\n\twant %q", len(data), data, content)
		}

		n, err = r.ReadAt(data[:1], int64(len(data)))
		if n != 0 || err != io.EOF {
			return fmt.Errorf("ReadAt(1, %d) = %v, %v, want 0, EOF", len(data), n, err)
		}

		for i := range data {
			data[i] = 0xfe
		}
		n, err = r.ReadAt(data[:cap(data)], 0)
		if n != len(data) || err != io.EOF {
			return fmt.Errorf("ReadAt(%d, 0) = %v, %v, want %d, EOF", cap(data), n, err, len(data))
		}
		if !bytes.Equal(data, content) {
			return fmt.Errorf("ReadAt(%d, 0) = %q\n\twant %q", len(data), data, content)
		}

		for i := range data {
			data[i] = 0xfe
		}
		for i := range data {
			n, err = r.ReadAt(data[i:i+1], int64(i))
			if n != 1 || err != nil && (i != len(data)-1 || err != io.EOF) {
				want := "nil"
				if i == len(data)-1 {
					want = "nil or EOF"
				}
				return fmt.Errorf("ReadAt(1, %d) = %v, %v, want 1, %s", i, n, err, want)
			}
			if data[i] != content[i] {
				return fmt.Errorf("ReadAt(1, %d) = %q want %q", i, data[i:i+1], content[i:i+1])
			}
		}
	}
	return nil
}
