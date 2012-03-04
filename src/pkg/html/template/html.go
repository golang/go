// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"fmt"
	"strings"
	"unicode/utf8"
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

// htmlReplacer returns s with runes replaced according to replacementTable
// and when badRunes is true, certain bad runes are allowed through unescaped.
func htmlReplacer(s string, replacementTable []string, badRunes bool) string {
	written, b := 0, new(bytes.Buffer)
	for i, r := range s {
		if int(r) < len(replacementTable) {
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
	s, c, i, allText := []byte(html), context{}, 0, true
	// Using the transition funcs helps us avoid mangling
	// `<div title="1>2">` or `I <3 Ponies!`.
	for i != len(s) {
		if c.delim == delimNone {
			st := c.state
			// Use RCDATA instead of parsing into JS or CSS styles.
			if c.element != elementNone && !isInTag(st) {
				st = stateRCDATA
			}
			d, nread := transitionFunc[st](c, s[i:])
			i1 := i + nread
			if c.state == stateText || c.state == stateRCDATA {
				// Emit text up to the start of the tag or comment.
				j := i1
				if d.state != c.state {
					for j1 := j - 1; j1 >= i; j1-- {
						if s[j1] == '<' {
							j = j1
							break
						}
					}
				}
				b.Write(s[i:j])
			} else {
				allText = false
			}
			c, i = d, i1
			continue
		}
		i1 := i + bytes.IndexAny(s[i:], delimEnds[c.delim])
		if i1 < i {
			break
		}
		if c.delim != delimSpaceOrTagEnd {
			// Consume any quote.
			i1++
		}
		c, i = context{state: stateTag, element: c.element}, i1
	}
	if allText {
		return html
	} else if c.state == stateText || c.state == stateRCDATA {
		b.Write(s[i:])
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
	if len(s) == 0 {
		// Avoid violation of structure preservation.
		// <input checked {{.K}}={{.V}}>.
		// Without this, if .K is empty then .V is the value of
		// checked, but otherwise .V is the value of the attribute
		// named .K.
		return filterFailsafe
	}
	s = strings.ToLower(s)
	if t := attrType(s); t != contentTypePlain {
		// TODO: Split attr and element name part filters so we can whitelist
		// attributes.
		return filterFailsafe
	}
	for _, r := range s {
		switch {
		case '0' <= r && r <= '9':
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
