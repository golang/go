// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Template library: default formatters

package template

import (
	"bytes"
	"fmt"
	"io"
)

// StringFormatter formats into the default string representation.
// It is stored under the name "str" and is the default formatter.
// You can override the default formatter by storing your default
// under the name "" in your custom formatter map.
func StringFormatter(w io.Writer, format string, value ...interface{}) {
	if len(value) == 1 {
		if b, ok := value[0].([]byte); ok {
			w.Write(b)
			return
		}
	}
	fmt.Fprint(w, value...)
}

var (
	esc_quot = []byte("&#34;") // shorter than "&quot;"
	esc_apos = []byte("&#39;") // shorter than "&apos;"
	esc_amp  = []byte("&amp;")
	esc_lt   = []byte("&lt;")
	esc_gt   = []byte("&gt;")
)

// HTMLEscape writes to w the properly escaped HTML equivalent
// of the plain text data s.
func HTMLEscape(w io.Writer, s []byte) {
	var esc []byte
	last := 0
	for i, c := range s {
		switch c {
		case '"':
			esc = esc_quot
		case '\'':
			esc = esc_apos
		case '&':
			esc = esc_amp
		case '<':
			esc = esc_lt
		case '>':
			esc = esc_gt
		default:
			continue
		}
		w.Write(s[last:i])
		w.Write(esc)
		last = i + 1
	}
	w.Write(s[last:])
}

// HTMLFormatter formats arbitrary values for HTML
func HTMLFormatter(w io.Writer, format string, value ...interface{}) {
	ok := false
	var b []byte
	if len(value) == 1 {
		b, ok = value[0].([]byte)
	}
	if !ok {
		var buf bytes.Buffer
		fmt.Fprint(&buf, value...)
		b = buf.Bytes()
	}
	HTMLEscape(w, b)
}
