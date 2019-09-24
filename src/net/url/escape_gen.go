// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

import (
	"bytes"
	"fmt"
	"go/format"
	"io/ioutil"
	"log"
)

type encoding int

// Keep in sync with src/net/url/url.go
const (
	encodePath encoding = 1 + iota
	encodePathSegment
	encodeHost
	encodeZone
	encodeUserPassword
	encodeQueryComponent
	encodeFragment

	_tableCount  // Used only inside this file
)

func main() {
	var unreservedCharactersTable = func() [256]bool {
		// §2.3 Unreserved characters
		// unreserved  = ALPHA / DIGIT / "-" / "." / "_" / "~"
		// Everything must be escaped excludes unreserved characters.
		var t [256]bool

		for i := 0; i < 256; i++ {
			t[i] = true
		}

		for i := '0'; i <= '9'; i++ {
			t[i] = false
		}

		for i := 'A'; i <= 'Z'; i++ {
			t[i] = false
		}

		for i := 'a'; i <= 'z'; i++ {
			t[i] = false
		}

		t['-'] = false
		t['.'] = false
		t['_'] = false
		t['~'] = false

		return t
	}()

	var shouldEscapePathTable = func() [256]bool {
		// §3.3 Path
		// The RFC allows : @ & = + $ but saves / ; , for assigning
		// meaning to individual path segments. This package
		// only manipulates the path as a whole, so we allow those
		// last three as well. That leaves only ? to escape.
		var t = unreservedCharactersTable

		for _, v := range `:@&=+$/;,`{
			t[v] = false
		}
		return t
	}()

	var shouldEscapePathSegmentTable = func() [256]bool {
		// §3.3 Path
		// The RFC allows : @ & = + $ but saves / ; , for assigning
		// meaning to individual path segments.
		var t = unreservedCharactersTable

		for _, v := range `:@&=+$` {
			t[v] = false
		}
		return t
	}()

	var shouldEscapeHostTable = func() [256]bool {
		// §3.2.2 Host allows
		//	sub-delims = "!" / "$" / "&" / "'" / "(" / ")" / "*" / "+" / "," / ";" / "="
		// as part of reg-name.
		// We add : because we include :port as part of host.
		// We add [ ] because we include [ipv6]:port as part of host.
		// We add < > because they're the only characters left that
		// we could possibly allow, and Parse will reject them if we
		// escape them (because hosts can't use %-encoding for
		// ASCII bytes).
		var t = unreservedCharactersTable

		for _, v := range `!$&'()*+,;=:[]<>"` {
			t[v] = false
		}
		return t
	}()

	var shouldEscapeZoneTable = shouldEscapeHostTable

	var shouldEscapeUserPasswordTable = func() [256]bool {
		// §3.2.1 User Information
		// The RFC allows ';', ':', '&', '=', '+', '$', and ',' in
		// userinfo, so we must escape only '@', '/', and '?'.
		// The parsing of userinfo treats ':' as special so we must escape
		// that too.
		var t = unreservedCharactersTable

		for _, v := range `;&=+$,` {
			t[v] = false
		}
		return t
	}()

	// The RFC reserves (so we must escape) everything.
	var shouldEscapeQueryComponentTable = unreservedCharactersTable

	var shouldEscapeFragmentTable = func() [256]bool {
		// § 3.5 Fragment
		//	fragment    = *( pchar / "/" / "?" )
		//	pchar       = unreserved / pct-encoded / sub-delims / ":" / "@"
		//
		// RFC 3986 §2.2 Reserved Characters
		// The RFC allows not escaping sub-delims. A subset of sub-delims are
		// included in reserved from RFC 2396 §2.2. The remaining sub-delims do not
		// need to be escaped. To minimize potential breakage, we apply two restrictions:
		// (1) we always escape sub-delims outside of the fragment, and (2) we always
		// escape single quote to avoid breaking callers that had previously assumed that
		// single quotes would be escaped. See issue #19917.
		var t = unreservedCharactersTable

		for _, v := range `$&+,/:;=?@!()*` {
			t[v] = false
		}
		return t
	}()

	// shouldEscapeTable concatenates all these tables into a string to speed up the conversion in the generated code
	shouldEscapeTable := make([]byte, _tableCount*256)

	encodes := []string{"", // Our encoding start with 1
		"shouldEscapePathTable", "shouldEscapePathSegmentTable",
		"shouldEscapeHostTable", "shouldEscapeZoneTable", "shouldEscapeUserPasswordTable",
		"shouldEscapeQueryComponentTable", "shouldEscapeFragmentTable"}
	encodesM := map[string][256]bool{
		"shouldEscapePathTable":           shouldEscapePathTable,
		"shouldEscapePathSegmentTable":    shouldEscapePathSegmentTable,
		"shouldEscapeHostTable":           shouldEscapeHostTable,
		"shouldEscapeZoneTable":           shouldEscapeZoneTable,
		"shouldEscapeUserPasswordTable":   shouldEscapeUserPasswordTable,
		"shouldEscapeQueryComponentTable": shouldEscapeQueryComponentTable,
		"shouldEscapeFragmentTable":       shouldEscapeFragmentTable}

	for i := 1; i < int(_tableCount); i++ {
		tableName := encodes[i]
		tableVar := encodesM[tableName]
		for j := 0; j < 256; j++ {
			if tableVar[j] {
				shouldEscapeTable[i*256+j] = 1
			}
		}
	}

	w := new(bytes.Buffer)
	w.WriteString(pre)
	fmt.Fprintf(w, table, string(shouldEscapeTable))

	out, err := format.Source(w.Bytes())
	if err != nil {
		log.Fatal(err)
	}

	if err := ioutil.WriteFile("escape.go", out, 0660); err != nil {
		log.Fatal(err)
	}

}

const pre = `// Code generated by go run escape_gen.go; DO NOT EDIT.

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package url

// Return true if the specified character should be escaped when
// appearing in a URL string, according to RFC 3986.
//
// Please be informed that for now shouldEscape does not check all
// reserved characters correctly. See golang.org/issue/5684.
// See src/net/url/escape_gen.go for more info information.
func shouldEscape(c byte, mode encoding) bool {
	if shouldEscapeTable[int(mode*256) + int(c)] == 0 {
		return false
	}
	return true
}
`

const table = `const shouldEscapeTable = %#v`
