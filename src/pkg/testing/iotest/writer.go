// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iotest

import (
	"io"
	"os"
)

// TruncateWriter returns a Writer that writes to w
// but stops silently after n bytes.
func TruncateWriter(w io.Writer, n int64) io.Writer {
	return &truncateWriter{w, n}
}

type truncateWriter struct {
	w io.Writer
	n int64
}

func (t *truncateWriter) Write(p []byte) (n int, err os.Error) {
	if t.n <= 0 {
		return len(p), nil
	}
	// real write
	n = len(p)
	if int64(n) > t.n {
		n = int(t.n)
	}
	n, err = t.w.Write(p[0:n])
	t.n -= int64(n)
	if err == nil {
		n = len(p)
	}
	return
}
