// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package htmlwriter

import (
	"os";
	"io";
	"fmt";
)

// Writer is a filter implementing the io.Write interface.
// It provides facilities to generate HTML tags and does
// HTML-escaping for text written through Write. Incoming
// text is assumed to be UTF-8 encoded.

export type Writer struct {
	// TODO should not export any of the fields
	writer io.Write;
}


func (b *Writer) Init(writer io.Write) *Writer {
	b.writer = writer;
	return b;
}


/* export */ func (p *Writer) Write(buf *[]byte) (written int, err *os.Error) {
	i0 := 0;
	for i := i0; i < len(buf); i++ {
		var s string;
		switch buf[i] {
		case '<': s = "&lt;";
		case '&': s = "&amp;";
		default: continue;
		}
		// write HTML escape instead of buf[i]
		w1, e1 := p.writer.Write(buf[i0 : i]);
		if e1 != nil {
			return i0 + w1, e1;
		}
		w2, e2 := io.WriteString(p.writer, s);
		if e2 != nil {
			return i0 + w1 /* not w2! */, e2;
		}
		i0 = i + 1;
	}
	written, err = p.writer.Write(buf[i0 : len(buf)]);
	return len(buf), err;
}


// ----------------------------------------------------------------------------
// HTML-specific interface

/* export */ func (p *Writer) Tag(s string) {
	// TODO proper error handling
	io.WriteString(p.writer, s);
}


// ----------------------------------------------------------------------------
//

export func New(writer io.Write) *Writer {
	return new(Writer).Init(writer);
}
