// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

import (
	"bytes"
	"strings"
	"unicode"
)

// ParseMediaType parses a media type value and any optional
// parameters, per RFC 1531.  Media types are the values in
// Content-Type and Content-Disposition headers (RFC 2183).  On
// success, ParseMediaType returns the media type converted to
// lowercase and trimmed of white space and a non-nil params.  On
// error, it returns an empty string and a nil params.
func ParseMediaType(v string) (mediatype string, params map[string]string) {
	i := strings.Index(v, ";")
	if i == -1 {
		i = len(v)
	}
	mediatype = strings.TrimSpace(strings.ToLower(v[0:i]))
	params = make(map[string]string)

	v = v[i:]
	for len(v) > 0 {
		v = strings.TrimLeftFunc(v, unicode.IsSpace)
		if len(v) == 0 {
			return
		}
		key, value, rest := consumeMediaParam(v)
		if key == "" {
			// Parse error.
			return "", nil
		}
		params[key] = value
		v = rest
	}
	return
}

func isNotTokenChar(rune int) bool {
	return !IsTokenChar(rune)
}

// consumeToken consumes a token from the beginning of provided
// string, per RFC 2045 section 5.1 (referenced from 2183), and return
// the token consumed and the rest of the string.  Returns ("", v) on
// failure to consume at least one character.
func consumeToken(v string) (token, rest string) {
	notPos := strings.IndexFunc(v, isNotTokenChar)
	if notPos == -1 {
		return v, ""
	}
	if notPos == 0 {
		return "", v
	}
	return v[0:notPos], v[notPos:]
}

// consumeValue consumes a "value" per RFC 2045, where a value is
// either a 'token' or a 'quoted-string'.  On success, consumeValue
// returns the value consumed (and de-quoted/escaped, if a
// quoted-string) and the rest of the string.  On failure, returns
// ("", v).
func consumeValue(v string) (value, rest string) {
	if !strings.HasPrefix(v, `"`) {
		return consumeToken(v)
	}

	// parse a quoted-string
	rest = v[1:] // consume the leading quote
	buffer := new(bytes.Buffer)
	var idx, rune int
	var nextIsLiteral bool
	for idx, rune = range rest {
		switch {
		case nextIsLiteral:
			if rune >= 0x80 {
				return "", v
			}
			buffer.WriteRune(rune)
			nextIsLiteral = false
		case rune == '"':
			return buffer.String(), rest[idx+1:]
		case IsQText(rune):
			buffer.WriteRune(rune)
		case rune == '\\':
			nextIsLiteral = true
		default:
			return "", v
		}
	}
	return "", v
}

func consumeMediaParam(v string) (param, value, rest string) {
	rest = strings.TrimLeftFunc(v, unicode.IsSpace)
	if !strings.HasPrefix(rest, ";") {
		return "", "", v
	}

	rest = rest[1:] // consume semicolon
	rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
	param, rest = consumeToken(rest)
	if param == "" {
		return "", "", v
	}
	if !strings.HasPrefix(rest, "=") {
		return "", "", v
	}
	rest = rest[1:] // consume equals sign
	value, rest = consumeValue(rest)
	if value == "" {
		return "", "", v
	}
	return param, value, rest
}
