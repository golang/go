// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Template library: default formatters

package template

import (
	"fmt";
	"io";
	"reflect";
)

// StringFormatter formats into the default string representation.
// It is stored under the name "str" and is the default formatter.
// You can override the default formatter by storing your default
// under the name "" in your custom formatter map.
func StringFormatter(w io.Write, value interface{}, format string) {
	fmt.Fprint(w, value);
}


var esc_amp = io.StringBytes("&amp;")
var esc_lt = io.StringBytes("&lt;")
var esc_gt = io.StringBytes("&gt;")

// HtmlEscape writes to w the properly escaped HTML equivalent
// of the plain text data s.
func HtmlEscape(w io.Write, s []byte) {
	last := 0;
	for i, c := range s {
		if c == '&' || c == '<' || c == '>' {
			w.Write(s[last:i]);
			switch c {
			case '&':
				w.Write(esc_amp);
			case '<':
				w.Write(esc_lt);
			case '>':
				w.Write(esc_gt);
			}
			last = i+1;
		}
	}
	w.Write(s[last:len(s)]);
}

// HtmlFormatter formats arbitrary values for HTML
func HtmlFormatter(w io.Write, value interface{}, format string) {
	var b io.ByteBuffer;
	fmt.Fprint(&b, value);
	HtmlEscape(w, b.Data());
}
