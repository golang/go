// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"strings"
	"utf8"
)

// unescapeEntity reads an entity like "&lt;" from b[src:] and writes the
// corresponding "<" to b[dst:], returning the incremented dst and src cursors.
// Precondition: src[0] == '&' && dst <= src.
func unescapeEntity(b []byte, dst, src int) (dst1, src1 int) {
	// TODO(nigeltao): Check that this entity substitution algorithm matches the spec:
	// http://www.whatwg.org/specs/web-apps/current-work/multipage/tokenization.html#consume-a-character-reference
	// TODO(nigeltao): Handle things like "&#20013;" or "&#x4e2d;".

	// i starts at 1 because we already know that s[0] == '&'.
	i, s := 1, b[src:]
	for i < len(s) {
		c := s[i]
		i++
		// Lower-cased characters are more common in entities, so we check for them first.
		if 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' {
			continue
		}
		if c != ';' {
			i--
		}
		x := entity[string(s[1:i])]
		if x != 0 {
			return dst + utf8.EncodeRune(x, b[dst:]), src + i
		}
		break
	}
	dst1, src1 = dst+i, src+i
	copy(b[dst:dst1], b[src:src1])
	return dst1, src1
}

// unescape unescapes b's entities in-place, so that "a&lt;b" becomes "a<b".
func unescape(b []byte) []byte {
	for i, c := range b {
		if c == '&' {
			dst, src := unescapeEntity(b, i, i)
			for src < len(b) {
				c := b[src]
				if c == '&' {
					dst, src = unescapeEntity(b, dst, src)
				} else {
					b[dst] = c
					dst, src = dst+1, src+1
				}
			}
			return b[0:dst]
		}
	}
	return b
}

// EscapeString escapes special characters like "<" to become "&lt;". It
// escapes only five such characters: amp, apos, lt, gt and quot.
// UnescapeString(EscapeString(s)) == s always holds, but the converse isn't
// always true.
func EscapeString(s string) string {
	// TODO(nigeltao): Do this much more efficiently.
	s = strings.Replace(s, `&`, `&amp;`, -1)
	s = strings.Replace(s, `'`, `&apos;`, -1)
	s = strings.Replace(s, `<`, `&lt;`, -1)
	s = strings.Replace(s, `>`, `&gt;`, -1)
	s = strings.Replace(s, `"`, `&quot;`, -1)
	return s
}

// UnescapeString unescapes entities like "&lt;" to become "<". It unescapes a
// larger range of entities than EscapeString escapes. For example, "&aacute;"
// unescapes to "á", as does "&#225;" and "&xE1;".
// UnescapeString(EscapeString(s)) == s always holds, but the converse isn't
// always true.
func UnescapeString(s string) string {
	for _, c := range s {
		if c == '&' {
			return string(unescape([]byte(s)))
		}
	}
	return s
}
