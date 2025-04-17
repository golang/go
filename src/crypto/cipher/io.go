// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher

import "io"

// The Stream* objects are so simple that all their members are public. Users
// can create them themselves.

// StreamReader wraps a [Stream] into an [io.Reader]. It calls XORKeyStream
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

// StreamWriter wraps a [Stream] into an io.Writer. It calls XORKeyStream
// to process each slice of data which passes through. If any [StreamWriter.Write]
// call returns short then the StreamWriter is out of sync and must be discarded.
// A StreamWriter has no internal buffering; [StreamWriter.Close] does not need
// to be called to flush write data.
type StreamWriter struct {
	S   Stream
	W   io.Writer
	Err error // unused
}

func (w StreamWriter) Write(src []byte) (n int, err error) {
	c := make([]byte, len(src))
	w.S.XORKeyStream(c, src)
	n, err = w.W.Write(c)
	if n != len(src) && err == nil { // should never happen
		err = io.ErrShortWrite
	}
	return
}

// Close closes the underlying Writer and returns its Close return value, if the Writer
// is also an io.Closer. Otherwise it returns nil.
func (w StreamWriter) Close() error {
	if c, ok := w.W.(io.Closer); ok {
		return c.Close()
	}
	return nil
}
