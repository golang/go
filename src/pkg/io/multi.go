// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io

import "os"

type multiReader struct {
	readers []Reader
}

func (mr *multiReader) Read(p []byte) (n int, err os.Error) {
	for len(mr.readers) > 0 {
		n, err = mr.readers[0].Read(p)
		if n > 0 || err != os.EOF {
			if err == os.EOF {
				// This shouldn't happen.
				// Well-behaved Readers should never
				// return non-zero bytes read with an
				// EOF.  But if so, we clean it.
				err = nil
			}
			return
		}
		mr.readers = mr.readers[1:]
	}
	return 0, os.EOF
}

// MultiReader returns a Reader that's the logical concatenation of
// the provided input readers.  They're read sequentially.  Once all
// inputs are drained, Read will return os.EOF.
func MultiReader(readers ...Reader) Reader {
	return &multiReader{readers}
}

type multiWriter struct {
	writers []Writer
}

func (t *multiWriter) Write(p []byte) (n int, err os.Error) {
	for _, w := range t.writers {
		n, err = w.Write(p)
		if err != nil {
			return
		}
		if n != len(p) {
			err = ErrShortWrite
			return
		}
	}
	return len(p), nil
}

// MultiWriter creates a writer that duplicates its writes to all the
// provided writers, similar to the Unix tee(1) command.
func MultiWriter(writers ...Writer) Writer {
	return &multiWriter{writers}
}
