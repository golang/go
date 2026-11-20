// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run gen_headermap.go

package httpcommon

import "net/textproto"

// LowerHeader returns the lowercase form of a header name,
// used on the wire for HTTP/2 and HTTP/3 requests.
func LowerHeader(v string) (lower string, ascii bool) {
	if s := lowerCommonHeader(v); s != "" {
		return s, true
	}
	return asciiToLower(v)
}

// CanonicalHeader canonicalizes a header name. (For example, "host" becomes "Host".)
func CanonicalHeader(v string) string {
	if s := canonCommonHeader(v); s != "" {
		return s
	}
	return textproto.CanonicalMIMEHeaderKey(v)
}

// CachedCanonicalHeader returns the canonical form of a well-known header name.
func CachedCanonicalHeader(v string) (string, bool) {
	s := canonCommonHeader(v)
	return s, s != ""
}
