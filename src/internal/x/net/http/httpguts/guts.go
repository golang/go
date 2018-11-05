// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package httpguts provides functions implementing various details
// of the HTTP specification.
//
// This package is shared by the standard library (which vendors it)
// and x/net/http2. It comes with no API stability promise.
package httpguts

import (
	"net/textproto"
	"strings"
)

// ValidTrailerHeader reports whether name is a valid header field name to appear
// in trailers.
// See RFC 7230, Section 4.1.2
func ValidTrailerHeader(name string) bool {
	name = textproto.CanonicalMIMEHeaderKey(name)
	if strings.HasPrefix(name, "If-") || badTrailer[name] {
		return false
	}
	return true
}

var badTrailer = map[string]bool{
	"Authorization":       true,
	"Cache-Control":       true,
	"Connection":          true,
	"Content-Encoding":    true,
	"Content-Length":      true,
	"Content-Range":       true,
	"Content-Type":        true,
	"Expect":              true,
	"Host":                true,
	"Keep-Alive":          true,
	"Max-Forwards":        true,
	"Pragma":              true,
	"Proxy-Authenticate":  true,
	"Proxy-Authorization": true,
	"Proxy-Connection":    true,
	"Range":               true,
	"Realm":               true,
	"Te":                  true,
	"Trailer":             true,
	"Transfer-Encoding":   true,
	"Www-Authenticate":    true,
}
