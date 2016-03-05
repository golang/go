// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"fmt"
	"io"
	"io/ioutil"
)

type Mode uint

// TODO(gri) These need a lot more work.

func ReadFile(filename string, mode Mode) (*File, error) {
	src, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return ReadBytes(src, mode)
}

func ReadBytes(src []byte, mode Mode) (*File, error) {
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

func Read(r io.Reader, mode Mode) (*File, error) {
	src, err := ioutil.ReadAll(r)
	if err != nil {
		return nil, err
	}
	return ReadBytes(src, mode)
}

func Write(w io.Writer, n *File) error {
	panic("unimplemented")
}
