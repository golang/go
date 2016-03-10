// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"fmt"
	"io"
	"os"
)

type Mode uint

// TODO(gri) These need a lot more work.

func ReadFile(filename string, mode Mode) (*File, error) {
	src, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	return Read(src, mode)
}

type bytesReader struct {
	data []byte
}

func (r *bytesReader) Read(p []byte) (int, error) {
	if len(r.data) > 0 {
		n := copy(p, r.data)
		r.data = r.data[n:]
		return n, nil
	}
	return 0, io.EOF
}

func ReadBytes(src []byte, mode Mode) (*File, error) {
	return Read(&bytesReader{src}, mode)
}

func Read(src io.Reader, mode Mode) (*File, error) {
	var p parser
	p.init(src)

	// skip initial BOM if present
	if p.getr() != '\ufeff' {
		p.ungetr()
	}

	p.next()
	ast := p.file()

	if nerrors > 0 {
		return nil, fmt.Errorf("%d syntax errors", nerrors)
	}

	return ast, nil
}

func Write(w io.Writer, n *File) error {
	panic("unimplemented")
}
