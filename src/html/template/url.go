// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"bytes"
	"fmt"
	"strings"
)

// urlFilter returns its input unless it contains an unsafe protocol in which
// case it defangs the entire URL.
func urlFilter(args ...interface{}) string {
	s, t := stringify(args...)
	if t == contentTypeURL {
		return s
	}
	if i := strings.IndexRune(s, ':'); i >= 0 && !strings.ContainsRune(s[:i], '/') {
		protocol := strings.ToLower(s[:i])
		if protocol != "http" && protocol != "https" && protocol != "mailto" {
			return "#" + filterFailsafe
		}
	}
	return s
}

// urlEscaper produces an output that can be embedded in a URL query.
// The output can be embedded in an HTML attribute without further escaping.
func urlEscaper(args ...interface{}) string {
	return urlProcessor(false, args...)
}

// urlEscaper normalizes URL content so it can be embedded in a quote-delimited
// string or parenthesis delimited url(...).
// The normalizer does not encode all HTML specials. Specifically, it does not
// encode '&' so correct embedding in an HTML attribute requires escaping of
// '&' to '&amp;'.
func urlNormalizer(args ...interface{}) string {
	return urlProcessor(true, args...)
}

// urlProcessor normalizes (when norm is true) or escapes its input to produce
// a valid hierarchical or opaque URL part.
func urlProcessor(norm bool, args ...interface{}) string {
	s, t := stringify(args...)
	if t == contentTypeURL {
		norm = true
	}
	var b bytes.Buffer
	written := 0
	// The byte loop below assumes that all URLs use UTF-8 as the
	// content-encoding. This is similar to the URI to IRI encoding scheme
	// defined in section 3.1 of  RFC 3987, and behaves the same as the
	// EcmaScript builtin encodeURIComponent.
	// It should not cause any misencoding of URLs in pages with
	// Content-type: text/html;charset=UTF-8.
	for i, n := 0, len(s); i < n; i++ {
		c := s[i]
		switch c {
		// Single quote and parens are sub-delims in RFC 3986, but we
		// escape them so the output can be embedded in single
		// quoted attributes and unquoted CSS url(...) constructs.
		// Single quotes are reserved in URLs, but are only used in
		// the obsolete "mark" rule in an appendix in RFC 3986
		// so can be safely encoded.
		case '!', '#', '$', '&', '*', '+', ',', '/', ':', ';', '=', '?', '@', '[', ']':
			if norm {
				continue
			}
		// Unreserved according to RFC 3986 sec 2.3
		// "For consistency, percent-encoded octets in the ranges of
		// ALPHA (%41-%5A and %61-%7A), DIGIT (%30-%39), hyphen (%2D),
		// period (%2E), underscore (%5F), or tilde (%7E) should not be
		// created by URI producers
		case '-', '.', '_', '~':
			continue
		case '%':
			// When normalizing do not re-encode valid escapes.
			if norm && i+2 < len(s) && isHex(s[i+1]) && isHex(s[i+2]) {
				continue
			}
		default:
			// Unreserved according to RFC 3986 sec 2.3
			if 'a' <= c && c <= 'z' {
				continue
			}
			if 'A' <= c && c <= 'Z' {
				continue
			}
			if '0' <= c && c <= '9' {
				continue
			}
		}
		b.WriteString(s[written:i])
		fmt.Fprintf(&b, "%%%02x", c)
		written = i + 1
	}
	if written == 0 {
		return s
	}
	b.WriteString(s[written:])
	return b.String()
}
