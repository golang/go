// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Encrypt/decrypt data by xor with a pseudo-random data stream.

package block

import (
	"io"
	"os"
)

// A dataStream is an interface to an unending stream of data,
// used by XorReader and XorWriter to model a pseudo-random generator.
// Calls to Next() return sequential blocks of data from the stream.
// Each call must return at least one byte: there is no EOF.
type dataStream interface {
	Next() []byte
}

type xorReader struct {
	r    io.Reader
	rand dataStream // pseudo-random
	buf  []byte     // data available from last call to rand
}

func newXorReader(rand dataStream, r io.Reader) io.Reader {
	x := new(xorReader)
	x.r = r
	x.rand = rand
	return x
}

func (x *xorReader) Read(p []byte) (n int, err os.Error) {
	n, err = x.r.Read(p)

	// xor input with stream.
	bp := 0
	buf := x.buf
	for i := 0; i < n; i++ {
		if bp >= len(buf) {
			buf = x.rand.Next()
			bp = 0
		}
		p[i] ^= buf[bp]
		bp++
	}
	x.buf = buf[bp:]
	return n, err
}

type xorWriter struct {
	w     io.Writer
	rand  dataStream // pseudo-random
	buf   []byte     // last buffer returned by rand
	extra []byte     // extra random data (use before buf)
	work  []byte     // work space
}

func newXorWriter(rand dataStream, w io.Writer) io.Writer {
	x := new(xorWriter)
	x.w = w
	x.rand = rand
	x.work = make([]byte, 4096)
	return x
}

func (x *xorWriter) Write(p []byte) (n int, err os.Error) {
	for len(p) > 0 {
		// Determine next chunk of random data
		// and xor with p into x.work.
		var chunk []byte
		m := len(p)
		if nn := len(x.extra); nn > 0 {
			// extra points into work, so edit directly
			if m > nn {
				m = nn
			}
			for i := 0; i < m; i++ {
				x.extra[i] ^= p[i]
			}
			chunk = x.extra[0:m]
		} else {
			// xor p ^ buf into work, refreshing buf as needed
			if nn := len(x.work); m > nn {
				m = nn
			}
			bp := 0
			buf := x.buf
			for i := 0; i < m; i++ {
				if bp >= len(buf) {
					buf = x.rand.Next()
					bp = 0
				}
				x.work[i] = buf[bp] ^ p[i]
				bp++
			}
			x.buf = buf[bp:]
			chunk = x.work[0:m]
		}

		// Write chunk.
		var nn int
		nn, err = x.w.Write(chunk)
		if nn != len(chunk) && err == nil {
			err = io.ErrShortWrite
		}
		if nn < len(chunk) {
			// Reconstruct the random bits from the unwritten
			// data and save them for next time.
			for i := nn; i < m; i++ {
				chunk[i] ^= p[i]
			}
			x.extra = chunk[nn:]
		}
		n += nn
		if err != nil {
			return
		}
		p = p[m:]
	}
	return
}
