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
	s, t := stringify(args...)
	if t == contentTypeHTML {
		return htmlReplacer(stripTags(s), htmlNospaceNormReplacementTable, false)
	}
	return htmlReplacer(s, htmlNospaceReplacementTable, false)
}

// attrEscaper escapes for inclusion in quoted attribute values.
func attrEscaper(args ...interface{}) string {
	s, t := stringify(args...)
	if t == contentTypeHTML {
		return htmlReplacer(stripTags(s), htmlNormReplacementTable, true)
	}
	return htmlReplacer(s, htmlReplacementTable, true)
}

// rcdataEscaper escapes for inclusion in an RCDATA element body.
func rcdataEscaper(args ...interface{}) string {
	s, t := stringify(args...)
	if t == contentTypeHTML {
		return htmlReplacer(s, htmlNormReplacementTable, true)
	}
	return htmlReplacer(s, htmlReplacementTable, true)
}

// htmlEscaper escapes for inclusion in HTML text.
func htmlEscaper(args ...interface{}) string {
	s, t := stringify(args...)
	if t == contentTypeHTML {
		return s
	}
	return htmlReplacer(s, htmlReplacementTable, true)
}

// htmlReplacementTable contains the runes that need to be escaped
// inside a quoted attribute value or in a text node.
var htmlReplacementTable = []string{
	// http://www.w3.org/TR/html5/tokenization.html#attribute-value-unquoted-state: "
	// U+0000 NULL Parse error. Append a U+FFFD REPLACEMENT
	// CHARACTER character to the current attribute's value.
	// "
	// and similarly
	// http://www.w3.org/TR/html5/tokenization.html#before-attribute-value-state
	0:    "\uFFFD",
	'"':  "&#34;",
	'&':  "&amp;",
	'\'': "&#39;",
	'+':  "&#43;",
	'<':  "&lt;",
	'>':  "&gt;",
}

// htmlNormReplacementTable is like htmlReplacementTable but without '&' to
// avoid over-encoding existing entities.
var htmlNormReplacementTable = []string{
	0:    "\uFFFD",
	'"':  "&#34;",
	'\'': "&#39;",
	'+':  "&#43;",
	'<':  "&lt;",
	'>':  "&gt;",
}

// htmlNospaceReplacementTable contains the runes that need to be escaped
// inside an unquoted attribute value.
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
var htmlNospaceReplacementTable = []string{
	0:    "&#xfffd;",
	'\t': "&#9;",
	'\n': "&#10;",
	'\v': "&#11;",
	'\f': "&#12;",
	'\r': "&#13;",
	' ':  "&#32;",
	'"':  "&#34;",
	'&':  "&amp;",
	'\'': "&#39;",
	'+':  "&#43;",
	'<':  "&lt;",
	'=':  "&#61;",
	'>':  "&gt;",
	// A parse error in the attribute value (unquoted) and 
	// before attribute value states.
	// Treated as a quoting character by IE.
	'`': "&#96;",
}

// htmlNospaceNormReplacementTable is like htmlNospaceReplacementTable but
// without '&' to avoid over-encoding existing entities.
var htmlNospaceNormReplacementTable = []string{
	0:    "&#xfffd;",
	'\t': "&#9;",
	'\n': "&#10;",
	'\v': "&#11;",
	'\f': "&#12;",
	'\r': "&#13;",
	' ':  "&#32;",
	'"':  "&#34;",
	'\'': "&#39;",
	'+':  "&#43;",
	'<':  "&lt;",
	'=':  "&#61;",
	'>':  "&gt;",
	// A parse error in the attribute value (unquoted) and 
	// before attribute value states.
	// Treated as a quoting character by IE.
	'`': "&#96;",
}

// htmlReplacer returns s with runes replaced acccording to replacementTable
// and when badRunes is true, certain bad runes are allowed through unescaped.
func htmlReplacer(s string, replacementTable []string, badRunes bool) string {
	written, b := 0, new(bytes.Buffer)
	for i, r := range s {
		if r < len(replacementTable) {
			if repl := replacementTable[r]; len(repl) != 0 {
				b.WriteString(s[written:i])
				b.WriteString(repl)
				// Valid as long as replacementTable doesn't 
				// include anything above 0x7f.
				written = i + utf8.RuneLen(r)
			}
		} else if badRunes {
			// No-op.
			// IE does not allow these ranges in unquoted attrs.
		} else if 0xfdd0 <= r && r <= 0xfdef || 0xfff0 <= r && r <= 0xffff {
			fmt.Fprintf(b, "%s&#x%x;", s[written:i], r)
			written = i + utf8.RuneLen(r)
		}
	}
	if written == 0 {
		return s
	}
	b.WriteString(s[written:])
	return b.String()
}

// stripTags takes a snippet of HTML and returns only the text content.
// For example, `<b>&iexcl;Hi!</b> <script>...</script>` -> `&iexcl;Hi! `.
func stripTags(html string) string {
	var b bytes.Buffer
	s, c := []byte(html), context{}
	// Using the transition funcs helps us avoid mangling
	// `<div title="1>2">` or `I <3 Ponies!`.
	for len(s) > 0 {
		if c.delim == delimNone {
			d, t := transitionFunc[c.state](c, s)
			if c.state == stateText || c.state == stateRCDATA {
				i := len(s) - len(t)
				// Emit text up to the start of the tag or comment.
				if d.state != c.state {
					for j := i - 1; j >= 0; j-- {
						if s[j] == '<' {
							i = j
							break
						}
					}
				}
				b.Write(s[:i])
			}
			c, s = d, t
			continue
		}
		i := bytes.IndexAny(s, delimEnds[c.delim])
		if i == -1 {
			break
		}
		if c.delim != delimSpaceOrTagEnd {
			// Consume any quote.
			i++
		}
		c, s = context{state: stateTag, element: c.element}, s[i:]
	}
	if c.state == stateText {
		if b.Len() == 0 {
			return html
		}
		b.Write(s)
	}
	return b.String()
}

// htmlNameFilter accepts valid parts of an HTML attribute or tag name or
// a known-safe HTML attribute.
func htmlNameFilter(args ...interface{}) string {
	s, t := stringify(args...)
	if t == contentTypeHTMLAttr {
		return s
	}
	for _, r := range s {
		switch {
		case '0' <= r && r <= '9':
		case 'A' <= r && r <= 'Z':
		case 'a' <= r && r <= 'z':
		default:
			return filterFailsafe
		}
	}
	return s
}

// commentEscaper returns the empty string regardless of input.
// Comment content does not correspond to any parsed structure or
// human-readable content, so the simplest and most secure policy is to drop
// content interpolated into comments.
// This approach is equally valid whether or not static comment content is
// removed from the template.
func commentEscaper(args ...interface{}) string {
	return ""
}
