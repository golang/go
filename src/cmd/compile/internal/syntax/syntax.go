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

// A Pragma value is a set of flags that augment a function
// declaration. Callers may assign meaning to the flags as
// appropriate.
type Pragma uint16

type ErrorHandler func(pos, line int, msg string)

// A PragmaHandler is used to process //line and //go: directives as
// they're scanned. The returned Pragma value will be unioned into the
// next FuncDecl node.
type PragmaHandler func(pos, line int, text string) Pragma

// TODO(gri) These need a lot more work.

func ReadFile(filename string, errh ErrorHandler, pragh PragmaHandler, mode Mode) (*File, error) {
	src, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer src.Close()
	return Read(src, errh, pragh, mode)
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

func ReadBytes(src []byte, errh ErrorHandler, pragh PragmaHandler, mode Mode) (*File, error) {
	return Read(&bytesReader{src}, errh, pragh, mode)
}

func Read(src io.Reader, errh ErrorHandler, pragh PragmaHandler, mode Mode) (*File, error) {
	var p parser
	p.init(src, errh, pragh)

	p.next()
	ast := p.file()

	if errh == nil && p.nerrors > 0 {
		return nil, fmt.Errorf("%d syntax errors", p.nerrors)
	}

	return ast, nil
}

func Write(w io.Writer, n *File) error {
	panic("unimplemented")
}
