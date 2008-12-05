// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package htmlwriter

import (
	"os";
	"io";
	"array";
	"utf8";
)

// Writer is a filter implementing the io.Write interface.
// It provides facilities to generate HTML tags and does
// proper HTML-escaping for text written through it.

export type Writer struct {
	// TODO should not export any of the fields
	writer io.Write;
}


func (b *Writer) Init(writer io.Write) *Writer {
	b.writer = writer;
	return b;
}


/* export */ func (b *Writer) Flush() *os.Error {
	return nil;
}


/* export */ func (b *Writer) Write(buf *[]byte) (written int, err *os.Error) {
	written, err = b.writer.Write(buf);  // BUG 6g - should just have return
	return written, err;
}


export func New(writer io.Write) *Writer {
	return new(Writer).Init(writer);
}
