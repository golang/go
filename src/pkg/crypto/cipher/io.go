// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher

import "io"

// The Stream* objects are so simple that all their members are public. Users
// can create them themselves.

// StreamReader wraps a Stream into an io.Reader. It calls XORKeyStream
// to process each slice of data which passes through.
type StreamReader struct {
	S Stream
	R io.Reader
}

func (r StreamReader) Read(dst []byte) (n int, err error) {
	n, err = r.R.Read(dst)
	r.S.XORKeyStream(dst[:n], dst[:n])
	return
}

// StreamWriter wraps a Stream into an io.Writer. It calls XORKeyStream
// to process each slice of data which passes through. If any Write call
// returns short then the StreamWriter is out of sync and must be discarded.
type StreamWriter struct {
	S   Stream
	W   io.Writer
	Err error
}

func (w StreamWriter) Write(src []byte) (n int, err error) {
	if w.Err != nil {
		return 0, w.Err
	}
	c := make([]byte, len(src))
	w.S.XORKeyStream(c, src)
	n, err = w.W.Write(c)
	if n != len(src) {
		if err == nil { // should never happen
			err = io.ErrShortWrite
		}
		w.Err = err
	}
	return
}

func (w StreamWriter) Close() error {
	// This saves us from either requiring a WriteCloser or having a
	// StreamWriterCloser.
	return w.W.(io.Closer).Close()
}
