// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"bytes"
	"strings"

	"encoding/json/jsontext"
)

// HTMLEscape appends to dst the JSON-encoded src with <, >, &, U+2028 and U+2029
// characters inside string literals changed to \u003c, \u003e, \u0026, \u2028, \u2029
// so that the JSON will be safe to embed inside HTML <script> tags.
// For historical reasons, web browsers don't honor standard HTML
// escaping within <script> tags, so an alternative JSON encoding must be used.
func HTMLEscape(dst *bytes.Buffer, src []byte) {
	dst.Grow(len(src))
	dst.Write(appendHTMLEscape(dst.AvailableBuffer(), src))
}

func appendHTMLEscape(dst, src []byte) []byte {
	const hex = "0123456789abcdef"
	// The characters can only appear in string literals,
	// so just scan the string one byte at a time.
	start := 0
	for i, c := range src {
		if c == '<' || c == '>' || c == '&' {
			dst = append(dst, src[start:i]...)
			dst = append(dst, '\\', 'u', '0', '0', hex[c>>4], hex[c&0xF])
			start = i + 1
		}
		// Convert U+2028 and U+2029 (E2 80 A8 and E2 80 A9).
		if c == 0xE2 && i+2 < len(src) && src[i+1] == 0x80 && src[i+2]&^1 == 0xA8 {
			dst = append(dst, src[start:i]...)
			dst = append(dst, '\\', 'u', '2', '0', '2', hex[src[i+2]&0xF])
			start = i + len("\u2029")
		}
	}
	return append(dst, src[start:]...)
}

// Compact appends to dst the JSON-encoded src with
// insignificant space characters elided.
func Compact(dst *bytes.Buffer, src []byte) error {
	dst.Grow(len(src))
	b := dst.AvailableBuffer()
	b, err := jsontext.AppendFormat(b, src,
		jsontext.AllowDuplicateNames(true),
		jsontext.AllowInvalidUTF8(true),
		jsontext.PreserveRawStrings(true))
	if err != nil {
		return transformSyntacticError(err)
	}
	dst.Write(b)
	return nil
}

// indentGrowthFactor specifies the growth factor of indenting JSON input.
// Empirically, the growth factor was measured to be between 1.4x to 1.8x
// for some set of compacted JSON with the indent being a single tab.
// Specify a growth factor slightly larger than what is observed
// to reduce probability of allocation in appendIndent.
// A factor no higher than 2 ensures that wasted space never exceeds 50%.
const indentGrowthFactor = 2

// Indent appends to dst an indented form of the JSON-encoded src.
// Each element in a JSON object or array begins on a new,
// indented line beginning with prefix followed by one or more
// copies of indent according to the indentation nesting.
// The data appended to dst does not begin with the prefix nor
// any indentation, to make it easier to embed inside other formatted JSON data.
// Although leading space characters (space, tab, carriage return, newline)
// at the beginning of src are dropped, trailing space characters
// at the end of src are preserved and copied to dst.
// For example, if src has no trailing spaces, neither will dst;
// if src ends in a trailing newline, so will dst.
func Indent(dst *bytes.Buffer, src []byte, prefix, indent string) error {
	dst.Grow(indentGrowthFactor * len(src))
	b := dst.AvailableBuffer()
	b, err := appendIndent(b, src, prefix, indent)
	dst.Write(b)
	return err
}

func appendIndent(dst, src []byte, prefix, indent string) ([]byte, error) {
	// In v2, only spaces and tabs are allowed, while v1 allowed any character.
	dstLen := len(dst)
	if len(strings.Trim(prefix, " \t"))+len(strings.Trim(indent, " \t")) > 0 {
		// Use placeholder spaces of correct length, and replace afterwards.
		invalidPrefix, invalidIndent := prefix, indent
		prefix = strings.Repeat(" ", len(prefix))
		indent = strings.Repeat(" ", len(indent))
		defer func() {
			b := dst[dstLen:]
			for i := bytes.IndexByte(b, '\n'); i >= 0; i = bytes.IndexByte(b, '\n') {
				b = b[i+len("\n"):]
				n := len(b) - len(bytes.TrimLeft(b, " ")) // len(prefix)+n*len(indent)
				spaces := b[:n]
				spaces = spaces[copy(spaces, invalidPrefix):]
				for len(spaces) > 0 {
					spaces = spaces[copy(spaces, invalidIndent):]
				}
				b = b[n:]
			}
		}()
	}

	dst, err := jsontext.AppendFormat(dst, src,
		jsontext.AllowDuplicateNames(true),
		jsontext.AllowInvalidUTF8(true),
		jsontext.PreserveRawStrings(true),
		jsontext.Multiline(true),
		jsontext.WithIndentPrefix(prefix),
		jsontext.WithIndent(indent))
	if err != nil {
		return dst[:dstLen], transformSyntacticError(err)
	}

	// In v2, trailing whitespace is discarded, while v1 preserved it.
	if n := len(src) - len(bytes.TrimRight(src, " \n\r\t")); n > 0 {
		dst = append(dst, src[len(src)-n:]...)
	}
	return dst, nil
}
