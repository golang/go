// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"utf8"
)

// entity is a map from HTML entity names to their values. The semicolon matters:
// http://www.whatwg.org/specs/web-apps/current-work/multipage/named-character-references.html
// lists both "amp" and "amp;" as two separate entries.
//
// TODO(nigeltao): Take the complete map from the HTML5 spec section 10.5 "Named character references".
// http://www.whatwg.org/specs/web-apps/current-work/multipage/named-character-references.html
// Note that the HTML5 list is larger than the HTML4 list at
// http://www.w3.org/TR/html4/sgml/entities.html
var entity = map[string]int{
	"aacute":  '\U000000E1',
	"aacute;": '\U000000E1',
	"amp;":    '\U00000026',
	"apos;":   '\U00000027',
	"gt;":     '\U0000003E',
	"lt;":     '\U0000003C',
	"quot;":   '\U00000022',
}

func init() {
	// We verify that the length of UTF-8 encoding of each value is <= 1 + len(key).
	// The +1 comes from the leading "&". This property implies that the length of
	// unescaped text is <= the length of escaped text.
	for k, v := range entity {
		if 1+len(k) < utf8.RuneLen(v) {
			panic("escaped entity &" + k + " is shorter than its UTF-8 encoding " + string(v))
		}
	}
}
