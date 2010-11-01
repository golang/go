// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parse URLs (actually URIs, but that seems overly pedantic).
// RFC 3986

package http

import (
	"os"
	"strconv"
	"strings"
)

// URLError reports an error and the operation and URL that caused it.
type URLError struct {
	Op    string
	URL   string
	Error os.Error
}

func (e *URLError) String() string { return e.Op + " " + e.URL + ": " + e.Error.String() }

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

type encoding int

const (
	encodePath encoding = 1 + iota
	encodeUserPassword
	encodeQueryComponent
	encodeFragment
	encodeOpaque
)


type URLEscapeError string

func (e URLEscapeError) String() string {
	return "invalid URL escape " + strconv.Quote(string(e))
}

// Return true if the specified character should be escaped when
// appearing in a URL string, according to RFC 2396.
// When 'all' is true the full range of reserved characters are matched.
func shouldEscape(c byte, mode encoding) bool {
	// RFC 2396 §2.3 Unreserved characters (alphanum)
	if 'A' <= c && c <= 'Z' || 'a' <= c && c <= 'z' || '0' <= c && c <= '9' {
		return false
	}
	switch c {
	case '-', '_', '.', '!', '~', '*', '\'', '(', ')': // §2.3 Unreserved characters (mark)
		return false

	case '$', '&', '+', ',', '/', ':', ';', '=', '?', '@': // §2.2 Reserved characters (reserved)
		// Different sections of the URL allow a few of
		// the reserved characters to appear unescaped.
		switch mode {
		case encodePath: // §3.3
			// The RFC allows : @ & = + $ , but saves / ; for assigning
			// meaning to individual path segments.  This package
			// only manipulates the path as a whole, so we allow those
			// last two as well.  Clients that need to distinguish between
			// `/foo;y=z/bar` and `/foo%3by=z/bar` will have to re-decode RawPath.
			// That leaves only ? to escape.
			return c == '?'

		case encodeUserPassword: // §3.2.2
			// The RFC allows ; : & = + $ , in userinfo, so we must escape only @ and /.
			// The parsing of userinfo treats : as special so we must escape that too.
			return c == '@' || c == '/' || c == ':'

		case encodeQueryComponent: // §3.4
			// The RFC reserves (so we must escape) everything.
			return true

		case encodeFragment: // §4.1
			// The RFC text is silent but the grammar allows
			// everything, so escape nothing.
			return false

		case encodeOpaque: // §3 opaque_part
			// The RFC allows opaque_part to use all characters
			// except that the leading / must be escaped.
			// (We implement that case in String.)
			return false
		}
	}

	// Everything else must be escaped.
	return true
}

// CanonicalPath applies the algorithm specified in RFC 2396 to
// simplify the path, removing unnecessary  . and .. elements.
func CanonicalPath(path string) string {
	buf := []byte(path)
	a := buf[0:0]
	// state helps to find /.. ^.. ^. and /. patterns.
	// state == 1 - prev char is '/' or beginning of the string.
	// state > 1  - prev state > 0 and prev char was '.'
	// state == 0 - otherwise
	state := 1
	cnt := 0
	for _, v := range buf {
		switch v {
		case '/':
			s := state
			state = 1
			switch s {
			case 2:
				a = a[0 : len(a)-1]
				continue
			case 3:
				if cnt > 0 {
					i := len(a) - 4
					for ; i >= 0 && a[i] != '/'; i-- {
					}
					a = a[0 : i+1]
					cnt--
					continue
				}
			default:
				if len(a) > 0 {
					cnt++
				}
			}
		case '.':
			if state > 0 {
				state++
			}
		default:
			state = 0
		}
		l := len(a)
		a = a[0 : l+1]
		a[l] = v
	}
	switch {
	case state == 2:
		a = a[0 : len(a)-1]
	case state == 3 && cnt > 0:
		i := len(a) - 4
		for ; i >= 0 && a[i] != '/'; i-- {
		}
		a = a[0 : i+1]
	}
	return string(a)
}

// URLUnescape unescapes a string in ``URL encoded'' form,
// converting %AB into the byte 0xAB and '+' into ' ' (space).
// It returns an error if any % is not followed
// by two hexadecimal digits.
// Despite the name, this encoding applies only to individual
// components of the query portion of the URL.
func URLUnescape(s string) (string, os.Error) {
	return urlUnescape(s, encodeQueryComponent)
}

// urlUnescape is like URLUnescape but mode specifies
// which section of the URL is being unescaped.
func urlUnescape(s string, mode encoding) (string, os.Error) {
	// Count %, check that they're well-formed.
	n := 0
	hasPlus := false
	for i := 0; i < len(s); {
		switch s[i] {
		case '%':
			n++
			if i+2 >= len(s) || !ishex(s[i+1]) || !ishex(s[i+2]) {
				s = s[i:]
				if len(s) > 3 {
					s = s[0:3]
				}
				return "", URLEscapeError(s)
			}
			i += 3
		case '+':
			hasPlus = mode == encodeQueryComponent
			i++
		default:
			i++
		}
	}

	if n == 0 && !hasPlus {
		return s, nil
	}

	t := make([]byte, len(s)-2*n)
	j := 0
	for i := 0; i < len(s); {
		switch s[i] {
		case '%':
			t[j] = unhex(s[i+1])<<4 | unhex(s[i+2])
			j++
			i += 3
		case '+':
			if mode == encodeQueryComponent {
				t[j] = ' '
			} else {
				t[j] = '+'
			}
			j++
			i++
		default:
			t[j] = s[i]
			j++
			i++
		}
	}
	return string(t), nil
}

// URLEscape converts a string into ``URL encoded'' form.
// Despite the name, this encoding applies only to individual
// components of the query portion of the URL.
func URLEscape(s string) string {
	return urlEscape(s, encodeQueryComponent)
}

func urlEscape(s string, mode encoding) string {
	spaceCount, hexCount := 0, 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		if shouldEscape(c, mode) {
			if c == ' ' && mode == encodeQueryComponent {
				spaceCount++
			} else {
				hexCount++
			}
		}
	}

	if spaceCount == 0 && hexCount == 0 {
		return s
	}

	t := make([]byte, len(s)+2*hexCount)
	j := 0
	for i := 0; i < len(s); i++ {
		switch c := s[i]; {
		case c == ' ' && mode == encodeQueryComponent:
			t[j] = '+'
			j++
		case shouldEscape(c, mode):
			t[j] = '%'
			t[j+1] = "0123456789abcdef"[c>>4]
			t[j+2] = "0123456789abcdef"[c&15]
			j += 3
		default:
			t[j] = s[i]
			j++
		}
	}
	return string(t)
}

// UnescapeUserinfo parses the RawUserinfo field of a URL
// as the form user or user:password and unescapes and returns
// the two halves.
//
// This functionality should only be used with legacy web sites.
// RFC 2396 warns that interpreting Userinfo this way
// ``is NOT RECOMMENDED, because the passing of authentication
// information in clear text (such as URI) has proven to be a
// security risk in almost every case where it has been used.''
func UnescapeUserinfo(rawUserinfo string) (user, password string, err os.Error) {
	u, p := split(rawUserinfo, ':', true)
	if user, err = urlUnescape(u, encodeUserPassword); err != nil {
		return "", "", err
	}
	if password, err = urlUnescape(p, encodeUserPassword); err != nil {
		return "", "", err
	}
	return
}

// EscapeUserinfo combines user and password in the form
// user:password (or just user if password is empty) and then
// escapes it for use as the URL.RawUserinfo field.
//
// This functionality should only be used with legacy web sites.
// RFC 2396 warns that interpreting Userinfo this way
// ``is NOT RECOMMENDED, because the passing of authentication
// information in clear text (such as URI) has proven to be a
// security risk in almost every case where it has been used.''
func EscapeUserinfo(user, password string) string {
	raw := urlEscape(user, encodeUserPassword)
	if password != "" {
		raw += ":" + urlEscape(password, encodeUserPassword)
	}
	return raw
}

// A URL represents a parsed URL (technically, a URI reference).
// The general form represented is:
//	scheme://[userinfo@]host/path[?query][#fragment]
// The Raw, RawAuthority, RawPath, and RawQuery fields are in "wire format"
// (special characters must be hex-escaped if not meant to have special meaning).
// All other fields are logical values; '+' or '%' represent themselves.
//
// The various Raw values are supplied in wire format because
// clients typically have to split them into pieces before further
// decoding.
type URL struct {
	Raw          string // the original string
	Scheme       string // scheme
	RawAuthority string // [userinfo@]host
	RawUserinfo  string // userinfo
	Host         string // host
	RawPath      string // /path[?query][#fragment]
	Path         string // /path
	OpaquePath   bool   // path is opaque (unrooted when scheme is present)
	RawQuery     string // query
	Fragment     string // fragment
}

// Maybe rawurl is of the form scheme:path.
// (Scheme must be [a-zA-Z][a-zA-Z0-9+-.]*)
// If so, return scheme, path; else return "", rawurl.
func getscheme(rawurl string) (scheme, path string, err os.Error) {
	for i := 0; i < len(rawurl); i++ {
		c := rawurl[i]
		switch {
		case 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z':
		// do nothing
		case '0' <= c && c <= '9' || c == '+' || c == '-' || c == '.':
			if i == 0 {
				return "", rawurl, nil
			}
		case c == ':':
			if i == 0 {
				return "", "", os.ErrorString("missing protocol scheme")
			}
			return rawurl[0:i], rawurl[i+1:], nil
		default:
			// we have encountered an invalid character,
			// so there is no valid scheme
			return "", rawurl, nil
		}
	}
	return "", rawurl, nil
}

// Maybe s is of the form t c u.
// If so, return t, c u (or t, u if cutc == true).
// If not, return s, "".
func split(s string, c byte, cutc bool) (string, string) {
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			if cutc {
				return s[0:i], s[i+1:]
			}
			return s[0:i], s[i:]
		}
	}
	return s, ""
}

// ParseURL parses rawurl into a URL structure.
// The string rawurl is assumed not to have a #fragment suffix.
// (Web browsers strip #fragment before sending the URL to a web server.)
func ParseURL(rawurl string) (url *URL, err os.Error) {
	if rawurl == "" {
		err = os.ErrorString("empty url")
		goto Error
	}
	url = new(URL)
	url.Raw = rawurl

	// Split off possible leading "http:", "mailto:", etc.
	// Cannot contain escaped characters.
	var path string
	if url.Scheme, path, err = getscheme(rawurl); err != nil {
		goto Error
	}

	if url.Scheme != "" && (len(path) == 0 || path[0] != '/') {
		// RFC 2396:
		// Absolute URI (has scheme) with non-rooted path
		// is uninterpreted.  It doesn't even have a ?query.
		// This is the case that handles mailto:name@example.com.
		url.RawPath = path

		if url.Path, err = urlUnescape(path, encodeOpaque); err != nil {
			goto Error
		}
		url.OpaquePath = true
	} else {
		// Split off query before parsing path further.
		url.RawPath = path
		path, query := split(path, '?', false)
		if len(query) > 1 {
			url.RawQuery = query[1:]
		}

		// Maybe path is //authority/path
		if url.Scheme != "" && len(path) > 2 && path[0:2] == "//" {
			url.RawAuthority, path = split(path[2:], '/', false)
			url.RawPath = url.RawPath[2+len(url.RawAuthority):]
		}

		// Split authority into userinfo@host.
		// If there's no @, split's default is wrong.  Check explicitly.
		var rawHost string
		if strings.Index(url.RawAuthority, "@") < 0 {
			rawHost = url.RawAuthority
		} else {
			url.RawUserinfo, rawHost = split(url.RawAuthority, '@', true)
		}

		// We leave RawAuthority only in raw form because clients
		// of common protocols should be using Userinfo and Host
		// instead.  Clients that wish to use RawAuthority will have to
		// interpret it themselves: RFC 2396 does not define the meaning.

		if strings.Contains(rawHost, "%") {
			// Host cannot contain escaped characters.
			err = os.ErrorString("hexadecimal escape in host")
			goto Error
		}
		url.Host = rawHost

		if url.Path, err = urlUnescape(path, encodePath); err != nil {
			goto Error
		}
	}
	return url, nil

Error:
	return nil, &URLError{"parse", rawurl, err}

}

// ParseURLReference is like ParseURL but allows a trailing #fragment.
func ParseURLReference(rawurlref string) (url *URL, err os.Error) {
	// Cut off #frag.
	rawurl, frag := split(rawurlref, '#', false)
	if url, err = ParseURL(rawurl); err != nil {
		return nil, err
	}
	url.Raw += frag
	url.RawPath += frag
	if len(frag) > 1 {
		frag = frag[1:]
		if url.Fragment, err = urlUnescape(frag, encodeFragment); err != nil {
			return nil, &URLError{"parse", rawurl, err}
		}
	}
	return url, nil
}

// String reassembles url into a valid URL string.
//
// There are redundant fields stored in the URL structure:
// the String method consults Scheme, Path, Host, RawUserinfo,
// RawQuery, and Fragment, but not Raw, RawPath or Authority.
func (url *URL) String() string {
	result := ""
	if url.Scheme != "" {
		result += url.Scheme + ":"
	}
	if url.Host != "" || url.RawUserinfo != "" {
		result += "//"
		if url.RawUserinfo != "" {
			// hide the password, if any
			info := url.RawUserinfo
			if i := strings.Index(info, ":"); i >= 0 {
				info = info[0:i] + ":******"
			}
			result += info + "@"
		}
		result += url.Host
	}
	if url.OpaquePath {
		path := url.Path
		if strings.HasPrefix(path, "/") {
			result += "%2f"
			path = path[1:]
		}
		result += urlEscape(path, encodeOpaque)
	} else {
		result += urlEscape(url.Path, encodePath)
	}
	if url.RawQuery != "" {
		result += "?" + url.RawQuery
	}
	if url.Fragment != "" {
		result += "#" + urlEscape(url.Fragment, encodeFragment)
	}
	return result
}
