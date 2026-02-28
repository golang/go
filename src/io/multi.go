// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io

type eofReader struct{}

func (eofReader) Read([]byte) (int, error) {
	return 0, EOF
}

type multiReader struct {
	readers []Reader
}

func (mr *multiReader) Read(p []byte) (n int, err error) {
	for len(mr.readers) > 0 {
		// Optimization to flatten nested multiReaders (Issue 13558).
		if len(mr.readers) == 1 {
			if r, ok := mr.readers[0].(*multiReader); ok {
				mr.readers = r.readers
				continue
			}
		}
		n, err = mr.readers[0].Read(p)
		if err == EOF {
			// Use eofReader instead of nil to avoid nil panic
			// after performing flatten (Issue 18232).
			mr.readers[0] = eofReader{} // permit earlier GC
			mr.readers = mr.readers[1:]
		}
		if n > 0 || err != EOF {
			if err == EOF && len(mr.readers) > 0 {
				// Don't return EOF yet. More readers remain.
				err = nil
			}
			return
		}
	}
	return 0, EOF
}

func (mr *multiReader) WriteTo(w Writer) (sum int64, err error) {
	return mr.writeToWithBuffer(w, make([]byte, 1024 * 32))
}

func (mr *multiReader) writeToWithBuffer(w Writer, buf []byte) (sum int64, err error) {
	for i, r := range mr.readers {
		var n int64
		if subMr, ok := r.(*multiReader); ok { // reuse buffer with nested multiReaders
			n, err = subMr.writeToWithBuffer(w, buf)
		} else {
			n, err = copyBuffer(w, r, buf)
		}
		sum += n
		if err != nil {
			mr.readers = mr.readers[i:] // permit resume / retry after error
			return sum, err
		}
		mr.readers[i] = nil // permit early GC
	}
	mr.readers = nil
	return sum, nil
}

var _ WriterTo = (*multiReader)(nil)

// MultiReader returns a Reader that's the logical concatenation of
// the provided input readers. They're read sequentially. Once all
// inputs have returned EOF, Read will return EOF.  If any of the readers
// return a non-nil, non-EOF error, Read will return that error.
func MultiReader(readers ...Reader) Reader {
	r := make([]Reader, len(readers))
	copy(r, readers)
	return &multiReader{r}
}

type multiWriter struct {
	writers []Writer
}

func (t *multiWriter) Write(p []byte) (n int, err error) {
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

var _ StringWriter = (*multiWriter)(nil)

func (t *multiWriter) WriteString(s string) (n int, err error) {
	var p []byte // lazily initialized if/when needed
	for _, w := range t.writers {
		if sw, ok := w.(StringWriter); ok {
			n, err = sw.WriteString(s)
		} else {
			if p == nil {
				p = []byte(s)
			}
			n, err = w.Write(p)
		}
		if err != nil {
			return
		}
		if n != len(s) {
			err = ErrShortWrite
			return
		}
	}
	return len(s), nil
}

// MultiWriter creates a writer that duplicates its writes to all the
// provided writers, similar to the Unix tee(1) command.
//
// Each write is written to each listed writer, one at a time.
// If a listed writer returns an error, that overall write operation
// stops and returns the error; it does not continue down the list.
func MultiWriter(writers ...Writer) Writer {
	allWriters := make([]Writer, 0, len(writers))
	for _, w := range writers {
		if mw, ok := w.(*multiWriter); ok {
			allWriters = append(allWriters, mw.writers...)
		} else {
			allWriters = append(allWriters, w)
		}
	}
	return &multiWriter{allWriters}
}
