// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filebuf

import (
	"bytes"
	"io"
)

// implement a Buf version from an io.Reader

type rbuf struct {
	buf          []byte // contents
	pos          int64
	seeks, reads int // number of calls. 0 seems right.
}

func (r *rbuf) Stats() Stat {
	return Stat{r.seeks, r.reads, int64(len(r.buf))}
}

func (r *rbuf) Size() int64 {
	return int64(len(r.buf))
}

// FromReader creates a Buf by copying the contents of an io.Reader
func FromReader(rd io.Reader) (Buf, error) {
	r := &rbuf{}
	x := bytes.NewBuffer(r.buf)
	_, err := io.Copy(x, rd)
	r.buf = x.Bytes()
	if err != nil {
		return nil, err
	}
	return r, nil
}

func (r *rbuf) Close() error {
	return nil
}

func (r *rbuf) Read(p []byte) (int, error) {
	n := copy(p, r.buf[r.pos:])
	r.pos += int64(n)
	if n == 0 || n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

func (r *rbuf) Seek(offset int64, whence int) (int64, error) {
	seekpos := offset
	switch whence {
	case io.SeekCurrent:
		seekpos += int64(r.pos)
	case io.SeekEnd:
		seekpos += int64(len(r.buf))
	}
	if seekpos < 0 || seekpos > int64(len(r.buf)) {
		if seekpos < 0 {
			r.pos = 0
			return 0, nil
		}
		r.pos = int64(len(r.buf))
		return r.pos, nil
	}
	r.pos = seekpos
	return seekpos, nil
}
