// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Code copied from x/net/http/httpguts/httplex.go
package auth

var isTokenTable = [256]bool{
	'!':  true,
	'#':  true,
	'$':  true,
	'%':  true,
	'&':  true,
	'\'': true,
	'*':  true,
	'+':  true,
	'-':  true,
	'.':  true,
	'0':  true,
	'1':  true,
	'2':  true,
	'3':  true,
	'4':  true,
	'5':  true,
	'6':  true,
	'7':  true,
	'8':  true,
	'9':  true,
	'A':  true,
	'B':  true,
	'C':  true,
	'D':  true,
	'E':  true,
	'F':  true,
	'G':  true,
	'H':  true,
	'I':  true,
	'J':  true,
	'K':  true,
	'L':  true,
	'M':  true,
	'N':  true,
	'O':  true,
	'P':  true,
	'Q':  true,
	'R':  true,
	'S':  true,
	'T':  true,
	'U':  true,
	'W':  true,
	'V':  true,
	'X':  true,
	'Y':  true,
	'Z':  true,
	'^':  true,
	'_':  true,
	'`':  true,
	'a':  true,
	'b':  true,
	'c':  true,
	'd':  true,
	'e':  true,
	'f':  true,
	'g':  true,
	'h':  true,
	'i':  true,
	'j':  true,
	'k':  true,
	'l':  true,
	'm':  true,
	'n':  true,
	'o':  true,
	'p':  true,
	'q':  true,
	'r':  true,
	's':  true,
	't':  true,
	'u':  true,
	'v':  true,
	'w':  true,
	'x':  true,
	'y':  true,
	'z':  true,
	'|':  true,
	'~':  true,
}

// isLWS reports whether b is linear white space, according
// to http://www.w3.org/Protocols/rfc2616/rfc2616-sec2.html#sec2.2
//
//	LWS            = [CRLF] 1*( SP | HT )
func isLWS(b byte) bool { return b == ' ' || b == '\t' }

// isCTL reports whether b is a control byte, according
// to http://www.w3.org/Protocols/rfc2616/rfc2616-sec2.html#sec2.2
//
//	CTL            = <any US-ASCII control character
//	                 (octets 0 - 31) and DEL (127)>
func isCTL(b byte) bool {
	const del = 0x7f // a CTL
	return b < ' ' || b == del
}

// validHeaderFieldName reports whether v is a valid HTTP/1.x header name.
// HTTP/2 imposes the additional restriction that uppercase ASCII
// letters are not allowed.
//
// RFC 7230 says:
//
//	header-field   = field-name ":" OWS field-value OWS
//	field-name     = token
//	token          = 1*tchar
//	tchar = "!" / "#" / "$" / "%" / "&" / "'" / "*" / "+" / "-" / "." /
//	        "^" / "_" / "`" / "|" / "~" / DIGIT / ALPHA
func validHeaderFieldName(v string) bool {
	if len(v) == 0 {
		return false
	}
	for i := 0; i < len(v); i++ {
		if !isTokenTable[v[i]] {
			return false
		}
	}
	return true
}

// validHeaderFieldValue reports whether v is a valid "field-value" according to
// http://www.w3.org/Protocols/rfc2616/rfc2616-sec4.html#sec4.2 :
//
//	message-header = field-name ":" [ field-value ]
//	field-value    = *( field-content | LWS )
//	field-content  = <the OCTETs making up the field-value
//	                 and consisting of either *TEXT or combinations
//	                 of token, separators, and quoted-string>
//
// http://www.w3.org/Protocols/rfc2616/rfc2616-sec2.html#sec2.2 :
//
//	TEXT           = <any OCTET except CTLs,
//	                  but including LWS>
//	LWS            = [CRLF] 1*( SP | HT )
//	CTL            = <any US-ASCII control character
//	                 (octets 0 - 31) and DEL (127)>
//
// RFC 7230 says:
//
//	field-value    = *( field-content / obs-fold )
//	obj-fold       =  N/A to http2, and deprecated
//	field-content  = field-vchar [ 1*( SP / HTAB ) field-vchar ]
//	field-vchar    = VCHAR / obs-text
//	obs-text       = %x80-FF
//	VCHAR          = "any visible [USASCII] character"
//
// http2 further says: "Similarly, HTTP/2 allows header field values
// that are not valid. While most of the values that can be encoded
// will not alter header field parsing, carriage return (CR, ASCII
// 0xd), line feed (LF, ASCII 0xa), and the zero character (NUL, ASCII
// 0x0) might be exploited by an attacker if they are translated
// verbatim. Any request or response that contains a character not
// permitted in a header field value MUST be treated as malformed
// (Section 8.1.2.6). Valid characters are defined by the
// field-content ABNF rule in Section 3.2 of [RFC7230]."
//
// This function does not (yet?) properly handle the rejection of
// strings that begin or end with SP or HTAB.
func validHeaderFieldValue(v string) bool {
	for i := 0; i < len(v); i++ {
		b := v[i]
		if isCTL(b) && !isLWS(b) {
			return false
		}
	}
	return true
}
