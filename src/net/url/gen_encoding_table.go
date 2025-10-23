// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

package main

import (
	"bytes"
	_ "embed"
	"fmt"
	"go/format"
	"io"
	"log"
	"maps"
	"os"
	"slices"
	"strconv"
	"strings"
)

// We embed this source file in the resulting code-generation program in order
// to extract the definitions of the encoding type and constants from it and
// include them in the generated file.
//
//go:embed gen_encoding_table.go
var genSource string

const filename = "encoding_table.go"

func main() {
	var out bytes.Buffer
	fmt.Fprintln(&out, "// Code generated from gen_encoding_table.go using 'go generate'; DO NOT EDIT.")
	fmt.Fprintln(&out)
	fmt.Fprintln(&out, "// Copyright 2025 The Go Authors. All rights reserved.")
	fmt.Fprintln(&out, "// Use of this source code is governed by a BSD-style")
	fmt.Fprintln(&out, "// license that can be found in the LICENSE file.")
	fmt.Fprintln(&out)
	fmt.Fprintln(&out, "package url")
	fmt.Fprintln(&out)
	generateEnc(&out, genSource)
	generateTable(&out)

	formatted, err := format.Source(out.Bytes())
	if err != nil {
		log.Fatal("format:", err)
	}

	err = os.WriteFile(filename, formatted, 0644)
	if err != nil {
		log.Fatal("WriteFile:", err)
	}
}

func generateEnc(w io.Writer, src string) {
	var writeLine bool
	for line := range strings.Lines(src) {
		if strings.HasPrefix(line, "// START encoding") {
			writeLine = true
			continue
		}
		if strings.HasPrefix(line, "// END encoding") {
			return
		}
		if writeLine {
			fmt.Fprint(w, line)
		}
	}
}

func generateTable(w io.Writer) {
	fmt.Fprintln(w, "var table = [256]encoding{")

	// Sort the encodings (in decreasing order) to guarantee a stable output.
	sortedEncs := slices.Sorted(maps.Keys(encNames))
	slices.Reverse(sortedEncs)

	for i := range 256 {
		c := byte(i)
		var lineBuf bytes.Buffer

		// Write key to line buffer.
		lineBuf.WriteString(strconv.QuoteRune(rune(c)))

		lineBuf.WriteByte(':')

		// Write value to line buffer.
		blankVal := true
		if ishex(c) {
			// Set the hexChar bit if this char is hexadecimal.
			lineBuf.WriteString("hexChar")
			blankVal = false
		}
		for _, enc := range sortedEncs {
			if !shouldEscape(c, enc) {
				if !blankVal {
					lineBuf.WriteByte('|')
				}
				// Set this encoding mode's bit if this char should NOT be
				// escaped.
				name := encNames[enc]
				lineBuf.WriteString(name)
				blankVal = false
			}
		}

		if !blankVal {
			lineBuf.WriteString(",\n")
			w.Write(lineBuf.Bytes())
		}
	}
	fmt.Fprintln(w, "}")
}

// START encoding (keep this marker comment in sync with genEnc)
type encoding uint8

const (
	encodePath encoding = 1 << iota
	encodePathSegment
	encodeHost
	encodeZone
	encodeUserPassword
	encodeQueryComponent
	encodeFragment

	// hexChar is actually NOT an encoding mode, but there are only seven
	// encoding modes. We might as well abuse the otherwise unused most
	// significant bit in uint8 to indicate whether a character is
	// hexadecimal.
	hexChar
)

// END encoding (keep this marker comment in sync with genEnc)

// Keep this in sync with the definitions of encoding mode constants.
var encNames = map[encoding]string{
	encodePath:           "encodePath",
	encodePathSegment:    "encodePathSegment",
	encodeHost:           "encodeHost",
	encodeZone:           "encodeZone",
	encodeUserPassword:   "encodeUserPassword",
	encodeQueryComponent: "encodeQueryComponent",
	encodeFragment:       "encodeFragment",
}

// Return true if the specified character should be escaped when
// appearing in a URL string, according to RFC 3986.
//
// Please be informed that for now shouldEscape does not check all
// reserved characters correctly. See golang.org/issue/5684.
func shouldEscape(c byte, mode encoding) bool {
	// §2.3 Unreserved characters (alphanum)
	if 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || '0' <= c && c <= '9' {
		return false
	}

	if mode == encodeHost || mode == encodeZone {
		// §3.2.2 Host allows
		//	sub-delims = "!" / "$" / "&" / "'" / "(" / ")" / "*" / "+" / "," / ";" / "="
		// as part of reg-name.
		// We add : because we include :port as part of host.
		// We add [ ] because we include [ipv6]:port as part of host.
		// We add < > because they're the only characters left that
		// we could possibly allow, and Parse will reject them if we
		// escape them (because hosts can't use %-encoding for
		// ASCII bytes).
		switch c {
		case '!', '$', '&', '\'', '(', ')', '*', '+', ',', ';', '=', ':', '[', ']', '<', '>', '"':
			return false
		}
	}

	switch c {
	case '-', '_', '.', '~': // §2.3 Unreserved characters (mark)
		return false

	case '$', '&', '+', ',', '/', ':', ';', '=', '?', '@': // §2.2 Reserved characters (reserved)
		// Different sections of the URL allow a few of
		// the reserved characters to appear unescaped.
		switch mode {
		case encodePath: // §3.3
			// The RFC allows : @ & = + $ but saves / ; , for assigning
			// meaning to individual path segments. This package
			// only manipulates the path as a whole, so we allow those
			// last three as well. That leaves only ? to escape.
			return c == '?'

		case encodePathSegment: // §3.3
			// The RFC allows : @ & = + $ but saves / ; , for assigning
			// meaning to individual path segments.
			return c == '/' || c == ';' || c == ',' || c == '?'

		case encodeUserPassword: // §3.2.1
			// The RFC allows ';', ':', '&', '=', '+', '$', and ',' in
			// userinfo, so we must escape only '@', '/', and '?'.
			// The parsing of userinfo treats ':' as special so we must escape
			// that too.
			return c == '@' || c == '/' || c == '?' || c == ':'

		case encodeQueryComponent: // §3.4
			// The RFC reserves (so we must escape) everything.
			return true

		case encodeFragment: // §4.1
			// The RFC text is silent but the grammar allows
			// everything, so escape nothing.
			return false
		}
	}

	if mode == encodeFragment {
		// RFC 3986 §2.2 allows not escaping sub-delims. A subset of sub-delims are
		// included in reserved from RFC 2396 §2.2. The remaining sub-delims do not
		// need to be escaped. To minimize potential breakage, we apply two restrictions:
		// (1) we always escape sub-delims outside of the fragment, and (2) we always
		// escape single quote to avoid breaking callers that had previously assumed that
		// single quotes would be escaped. See issue #19917.
		switch c {
		case '!', '(', ')', '*':
			return false
		}
	}

	// Everything else must be escaped.
	return true
}

func ishex(c byte) bool {
	return '0' <= c && c <= '9' ||
		'a' <= c && c <= 'f' ||
		'A' <= c && c <= 'F'
}
