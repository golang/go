// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

import (
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
	"unicode"
)

// FormatMediaType serializes mediatype t and the parameters
// param as a media type conforming to RFC 2045 and RFC 2616.
// The type and parameter names are written in lower-case.
// When any of the arguments result in a standard violation then
// FormatMediaType returns the empty string.
func FormatMediaType(t string, param map[string]string) string {
	var b strings.Builder
	if major, sub, ok := strings.Cut(t, "/"); !ok {
		if !isToken(t) {
			return ""
		}
		b.WriteString(strings.ToLower(t))
	} else {
		if !isToken(major) || !isToken(sub) {
			return ""
		}
		b.WriteString(strings.ToLower(major))
		b.WriteByte('/')
		b.WriteString(strings.ToLower(sub))
	}

	for _, attribute := range slices.Sorted(maps.Keys(param)) {
		value := param[attribute]
		b.WriteByte(';')
		b.WriteByte(' ')
		if !isToken(attribute) {
			return ""
		}
		b.WriteString(strings.ToLower(attribute))

		needEnc := needsEncoding(value)
		if needEnc {
			// RFC 2231 section 4
			b.WriteByte('*')
		}
		b.WriteByte('=')

		if needEnc {
			b.WriteString("utf-8''")

			offset := 0
			for index := 0; index < len(value); index++ {
				ch := value[index]
				// {RFC 2231 section 7}
				// attribute-char := <any (US-ASCII) CHAR except SPACE, CTLs, "*", "'", "%", or tspecials>
				if ch <= ' ' || ch >= 0x7F ||
					ch == '*' || ch == '\'' || ch == '%' ||
					isTSpecial(ch) {

					b.WriteString(value[offset:index])
					offset = index + 1

					b.WriteByte('%')
					b.WriteByte(upperhex[ch>>4])
					b.WriteByte(upperhex[ch&0x0F])
				}
			}
			b.WriteString(value[offset:])
			continue
		}

		if isToken(value) {
			b.WriteString(value)
			continue
		}

		b.WriteByte('"')
		offset := 0
		for index := 0; index < len(value); index++ {
			character := value[index]
			if character == '"' || character == '\\' {
				b.WriteString(value[offset:index])
				offset = index
				b.WriteByte('\\')
			}
		}
		b.WriteString(value[offset:])
		b.WriteByte('"')
	}
	return b.String()
}

func checkMediaTypeDisposition(s string) error {
	typ, rest := consumeToken(s)
	if typ == "" {
		return errors.New("mime: no media type")
	}
	if rest == "" {
		return nil
	}
	var ok bool
	if rest, ok = strings.CutPrefix(rest, "/"); !ok {
		return errors.New("mime: expected slash after first token")
	}
	subtype, rest := consumeToken(rest)
	if subtype == "" {
		return errors.New("mime: expected token after slash")
	}
	if rest != "" {
		return errors.New("mime: unexpected content after media subtype")
	}
	return nil
}

// ErrInvalidMediaParameter is returned by [ParseMediaType] if
// the media type value was found but there was an error parsing
// the optional parameters
var ErrInvalidMediaParameter = errors.New("mime: invalid media parameter")

// ParseMediaType parses a media type value and any optional
// parameters, per RFC 1521.  Media types are the values in
// Content-Type and Content-Disposition headers (RFC 2183).
// On success, ParseMediaType returns the media type converted
// to lowercase and trimmed of white space and a non-nil map.
// If there is an error parsing the optional parameter,
// the media type will be returned along with the error
// [ErrInvalidMediaParameter].
// The returned map, params, maps from the lowercase
// attribute to the attribute value with its case preserved.
func ParseMediaType(v string) (mediatype string, params map[string]string, err error) {
	base, _, _ := strings.Cut(v, ";")
	mediatype = strings.TrimSpace(strings.ToLower(base))

	err = checkMediaTypeDisposition(mediatype)
	if err != nil {
		return "", nil, err
	}

	params = make(map[string]string)

	// Map of base parameter name -> parameter name -> value
	// for parameters containing a '*' character.
	// Lazily initialized.
	var continuation map[string]map[string]string

	v = v[len(base):]
	for len(v) > 0 {
		v = strings.TrimLeftFunc(v, unicode.IsSpace)
		if len(v) == 0 {
			break
		}
		key, value, rest := consumeMediaParam(v)
		if key == "" {
			if strings.TrimSpace(rest) == ";" {
				// Ignore trailing semicolons.
				// Not an error.
				break
			}
			// Parse error.
			return mediatype, nil, ErrInvalidMediaParameter
		}

		pmap := params
		if baseName, _, ok := strings.Cut(key, "*"); ok {
			if continuation == nil {
				continuation = make(map[string]map[string]string)
			}
			if pmap, ok = continuation[baseName]; !ok {
				continuation[baseName] = make(map[string]string)
				pmap = continuation[baseName]
			}
		}
		if v, exists := pmap[key]; exists && v != value {
			// Duplicate parameter names are incorrect, but we allow them if they are equal.
			return "", nil, errors.New("mime: duplicate parameter name")
		}
		pmap[key] = value
		v = rest
	}

	// Stitch together any continuations or things with stars
	// (i.e. RFC 2231 things with stars: "foo*0" or "foo*")
	var buf strings.Builder
	for key, pieceMap := range continuation {
		singlePartKey := key + "*"
		if v, ok := pieceMap[singlePartKey]; ok {
			if decv, ok := decode2231Enc(v); ok {
				params[key] = decv
			}
			continue
		}

		buf.Reset()
		valid := false
		for n := 0; ; n++ {
			simplePart := fmt.Sprintf("%s*%d", key, n)
			if v, ok := pieceMap[simplePart]; ok {
				valid = true
				buf.WriteString(v)
				continue
			}
			encodedPart := simplePart + "*"
			v, ok := pieceMap[encodedPart]
			if !ok {
				break
			}
			valid = true
			if n == 0 {
				if decv, ok := decode2231Enc(v); ok {
					buf.WriteString(decv)
				}
			} else {
				decv, _ := percentHexUnescape(v)
				buf.WriteString(decv)
			}
		}
		if valid {
			params[key] = buf.String()
		}
	}

	return
}

func decode2231Enc(v string) (string, bool) {
	charset, v, ok := strings.Cut(v, "'")
	if !ok {
		return "", false
	}
	// TODO: ignoring the language part for now. If anybody needs it, we'll
	// need to decide how to expose it in the API. But I'm not sure
	// anybody uses it in practice.
	_, extOtherVals, ok := strings.Cut(v, "'")
	if !ok {
		return "", false
	}
	charset = strings.ToLower(charset)
	switch charset {
	case "us-ascii", "utf-8":
	default:
		// Empty or unsupported encoding.
		return "", false
	}
	return percentHexUnescape(extOtherVals)
}

// consumeToken consumes a token from the beginning of provided
// string, per RFC 2045 section 5.1 (referenced from 2183), and return
// the token consumed and the rest of the string. Returns ("", v) on
// failure to consume at least one character.
func consumeToken(v string) (token, rest string) {
	for i := range len(v) {
		if !isTokenChar(v[i]) {
			return v[:i], v[i:]
		}
	}
	return v, ""
}

// consumeValue consumes a "value" per RFC 2045, where a value is
// either a 'token' or a 'quoted-string'.  On success, consumeValue
// returns the value consumed (and de-quoted/escaped, if a
// quoted-string) and the rest of the string. On failure, returns
// ("", v).
func consumeValue(v string) (value, rest string) {
	if v == "" {
		return
	}
	if v[0] != '"' {
		return consumeToken(v)
	}

	// parse a quoted-string
	buffer := new(strings.Builder)
	for i := 1; i < len(v); i++ {
		r := v[i]
		if r == '"' {
			return buffer.String(), v[i+1:]
		}
		// When MSIE sends a full file path (in "intranet mode"), it does not
		// escape backslashes: "C:\dev\go\foo.txt", not "C:\\dev\\go\\foo.txt".
		//
		// No known MIME generators emit unnecessary backslash escapes
		// for simple token characters like numbers and letters.
		//
		// If we see an unnecessary backslash escape, assume it is from MSIE
		// and intended as a literal backslash. This makes Go servers deal better
		// with MSIE without affecting the way they handle conforming MIME
		// generators.
		if r == '\\' && i+1 < len(v) && isTSpecial(v[i+1]) {
			buffer.WriteByte(v[i+1])
			i++
			continue
		}
		if r == '\r' || r == '\n' {
			return "", v
		}
		buffer.WriteByte(v[i])
	}
	// Did not find end quote.
	return "", v
}

func consumeMediaParam(v string) (param, value, rest string) {
	rest = strings.TrimLeftFunc(v, unicode.IsSpace)
	var ok bool
	if rest, ok = strings.CutPrefix(rest, ";"); !ok {
		return "", "", v
	}

	rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
	param, rest = consumeToken(rest)
	param = strings.ToLower(param)
	if param == "" {
		return "", "", v
	}

	rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
	if rest, ok = strings.CutPrefix(rest, "="); !ok {
		return "", "", v
	}
	rest = strings.TrimLeftFunc(rest, unicode.IsSpace)
	value, rest2 := consumeValue(rest)
	if value == "" && rest2 == rest {
		return "", "", v
	}
	rest = rest2
	return param, value, rest
}

func percentHexUnescape(s string) (string, bool) {
	// Count %, check that they're well-formed.
	percents := 0
	for i := 0; i < len(s); {
		if s[i] != '%' {
			i++
			continue
		}
		percents++
		if i+2 >= len(s) || !ishex(s[i+1]) || !ishex(s[i+2]) {
			return "", false
		}
		i += 3
	}
	if percents == 0 {
		return s, true
	}

	t := make([]byte, len(s)-2*percents)
	j := 0
	for i := 0; i < len(s); {
		switch s[i] {
		case '%':
			t[j] = unhex(s[i+1])<<4 | unhex(s[i+2])
			j++
			i += 3
		default:
			t[j] = s[i]
			j++
			i++
		}
	}
	return string(t), true
}

func ishex(c byte) bool {
	switch {
	case '0' <= c && c <= '9':
		return true
	case 'a' <= c && c <= 'f':
		return true
	case 'A' <= c && c <= 'F':
		return true
	}
	return false
}

func unhex(c byte) byte {
	switch {
	case '0' <= c && c <= '9':
		return c - '0'
	case 'a' <= c && c <= 'f':
		return c - 'a' + 10
	case 'A' <= c && c <= 'F':
		return c - 'A' + 10
	}
	return 0
}
