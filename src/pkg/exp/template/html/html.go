// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bytes"
	"fmt"
	"utf8"
)

// htmlNospaceEscaper escapes for inclusion in unquoted attribute values.
func htmlNospaceEscaper(args ...interface{}) string {
	s := stringify(args...)
	// The set of runes escaped is the union of the HTML specials and
	// those determined by running the JS below in browsers:

	// <div id=d></div>
	// <script>(function () {
	// var a = [], d = document.getElementById("d"), i, c, s;
	// for (i = 0; i < 0x10000; ++i) {
	//   c = String.fromCharCode(i);
	//   d.innerHTML = "<span title=" + c + "lt" + c + "></span>"
	//   s = d.getElementsByTagName("SPAN")[0];
	//   if (!s || s.title !== c + "lt" + c) { a.push(i.toString(16)); }
	// }
	// document.write(a.join(", "));
	// })()</script>

	var b bytes.Buffer
	written := 0
	for i, r := range s {
		var repl string
		switch r {
		case 0:
			// http://www.w3.org/TR/html5/tokenization.html#attribute-value-unquoted-state: "
			// U+0000 NULL Parse error. Append a U+FFFD REPLACEMENT
			// CHARACTER character to the current attribute's value.
			// "
			// and similarly
			// http://www.w3.org/TR/html5/tokenization.html#before-attribute-value-state
			repl = "\uFFFD"
		case '\t':
			repl = "&#9;"
		case '\n':
			repl = "&#10;"
		case '\v':
			repl = "&#11;"
		case '\f':
			repl = "&#12;"
		case '\r':
			repl = "&#13;"
		case ' ':
			repl = "&#32;"
		case '"':
			repl = "&#34;"
		case '&':
			repl = "&amp;"
		case '\'':
			repl = "&#39;"
		case '+':
			repl = "&#43;"
		case '<':
			repl = "&lt;"
		case '=':
			repl = "&#61;"
		case '>':
			repl = "&gt;"
		case '`':
			// A parse error in the attribute value (unquoted) and 
			// before attribute value states.
			// Treated as a quoting character by IE.
			repl = "&#96;"
		default:
			// IE does not allow the ranges below raw in attributes.
			if 0xfdd0 <= r && r <= 0xfdef || 0xfff0 <= r && r <= 0xffff {
				b.WriteString(s[written:i])
				b.WriteString("&#x")
				b.WriteByte("0123456789abcdef"[r>>24])
				b.WriteByte("0123456789abcdef"[r>>16&0xf])
				b.WriteByte("0123456789abcdef"[r>>8&0xf])
				b.WriteByte("0123456789abcdef"[r&0xf])
				b.WriteByte(';')
				fmt.Fprintf(&b, "&#x%x;", r)
				written = i + utf8.RuneLen(r)
			}
			continue
		}
		b.WriteString(s[written:i])
		b.WriteString(repl)
		// Valid as long as we don't include any cases above in the
		// 0x80-0xff range.
		written = i + utf8.RuneLen(r)
	}
	if written == 0 {
		return s
	}
	b.WriteString(s[written:])
	return b.String()
}
