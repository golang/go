// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package URL parses URLs and implements query escaping.
// See RFC 3986.
package url

import (
	"os"
	"strconv"
	"strings"
)

// Error reports an error and the operation and URL that caused it.
type Error struct {
	Op    string
	URL   string
	Error os.Error
}

func (e *Error) String() string { return e.Op + " " + e.URL + ": " + e.Error.String() }

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

type EscapeError string

func (e EscapeError) String() string {
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

// QueryUnescape does the inverse transformation of QueryEscape, converting
// %AB into the byte 0xAB and '+' into ' ' (space). It returns an error if
// any % is not followed by two hexadecimal digits.
func QueryUnescape(s string) (string, os.Error) {
	return unescape(s, encodeQueryComponent)
}

// unescape unescapes a string; the mode specifies
// which section of the URL string is being unescaped.
func unescape(s string, mode encoding) (string, os.Error) {
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
				return "", EscapeError(s)
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

// QueryEscape escapes the string so it can be safely placed
// inside a URL query.
func QueryEscape(s string) string {
	return escape(s, encodeQueryComponent)
}

func escape(s string, mode encoding) string {
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
			t[j+1] = "0123456789ABCDEF"[c>>4]
			t[j+2] = "0123456789ABCDEF"[c&15]
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
	if user, err = unescape(u, encodeUserPassword); err != nil {
		return "", "", err
	}
	if password, err = unescape(p, encodeUserPassword); err != nil {
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
	raw := escape(user, encodeUserPassword)
	if password != "" {
		raw += ":" + escape(password, encodeUserPassword)
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
				return "", "", os.NewError("missing protocol scheme")
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

// Parse parses rawurl into a URL structure.
// The string rawurl is assumed not to have a #fragment suffix.
// (Web browsers strip #fragment before sending the URL to a web server.)
// The rawurl may be relative or absolute.
func Parse(rawurl string) (url *URL, err os.Error) {
	return parse(rawurl, false)
}

// ParseRequest parses rawurl into a URL structure.  It assumes that
// rawurl was received from an HTTP request, so the rawurl is interpreted
// only as an absolute URI or an absolute path.
// The string rawurl is assumed not to have a #fragment suffix.
// (Web browsers strip #fragment before sending the URL to a web server.)
func ParseRequest(rawurl string) (url *URL, err os.Error) {
	return parse(rawurl, true)
}

// parse parses a URL from a string in one of two contexts.  If
// viaRequest is true, the URL is assumed to have arrived via an HTTP request,
// in which case only absolute URLs or path-absolute relative URLs are allowed.
// If viaRequest is false, all forms of relative URLs are allowed.
func parse(rawurl string, viaRequest bool) (url *URL, err os.Error) {
	var (
		leadingSlash bool
		path         string
	)

	if rawurl == "" {
		err = os.NewError("empty url")
		goto Error
	}
	url = new(URL)
	url.Raw = rawurl

	// Split off possible leading "http:", "mailto:", etc.
	// Cannot contain escaped characters.
	if url.Scheme, path, err = getscheme(rawurl); err != nil {
		goto Error
	}
	leadingSlash = strings.HasPrefix(path, "/")

	if url.Scheme != "" && !leadingSlash {
		// RFC 2396:
		// Absolute URI (has scheme) with non-rooted path
		// is uninterpreted.  It doesn't even have a ?query.
		// This is the case that handles mailto:name@example.com.
		url.RawPath = path

		if url.Path, err = unescape(path, encodeOpaque); err != nil {
			goto Error
		}
		url.OpaquePath = true
	} else {
		if viaRequest && !leadingSlash {
			err = os.NewError("invalid URI for request")
			goto Error
		}

		// Split off query before parsing path further.
		url.RawPath = path
		path, query := split(path, '?', false)
		if len(query) > 1 {
			url.RawQuery = query[1:]
		}

		// Maybe path is //authority/path
		if (url.Scheme != "" || !viaRequest) &&
			strings.HasPrefix(path, "//") && !strings.HasPrefix(path, "///") {
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
			err = os.NewError("hexadecimal escape in host")
			goto Error
		}
		url.Host = rawHost

		if url.Path, err = unescape(path, encodePath); err != nil {
			goto Error
		}
	}
	return url, nil

Error:
	return nil, &Error{"parse", rawurl, err}

}

// ParseWithReference is like Parse but allows a trailing #fragment.
func ParseWithReference(rawurlref string) (url *URL, err os.Error) {
	// Cut off #frag.
	rawurl, frag := split(rawurlref, '#', false)
	if url, err = Parse(rawurl); err != nil {
		return nil, err
	}
	url.Raw += frag
	url.RawPath += frag
	if len(frag) > 1 {
		frag = frag[1:]
		if url.Fragment, err = unescape(frag, encodeFragment); err != nil {
			return nil, &Error{"parse", rawurl, err}
		}
	}
	return url, nil
}

// String reassembles url into a valid URL string.
//
// There are redundant fields stored in the URL structure:
// the String method consults Scheme, Path, Host, RawUserinfo,
// RawQuery, and Fragment, but not Raw, RawPath or RawAuthority.
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
		result += escape(path, encodeOpaque)
	} else {
		result += escape(url.Path, encodePath)
	}
	if url.RawQuery != "" {
		result += "?" + url.RawQuery
	}
	if url.Fragment != "" {
		result += "#" + escape(url.Fragment, encodeFragment)
	}
	return result
}

// Values maps a string key to a list of values.
// It is typically used for query parameters and form values.
// Unlike in the http.Header map, the keys in a Values map
// are case-sensitive.
type Values map[string][]string

// Get gets the first value associated with the given key.
// If there are no values associated with the key, Get returns
// the empty string. To access multiple values, use the map
// directly.
func (v Values) Get(key string) string {
	if v == nil {
		return ""
	}
	vs, ok := v[key]
	if !ok || len(vs) == 0 {
		return ""
	}
	return vs[0]
}

// Set sets the key to value. It replaces any existing
// values.
func (v Values) Set(key, value string) {
	v[key] = []string{value}
}

// Add adds the key to value. It appends to any existing
// values associated with key.
func (v Values) Add(key, value string) {
	v[key] = append(v[key], value)
}

// Del deletes the values associated with key.
func (v Values) Del(key string) {
	delete(v, key)
}

// ParseQuery parses the URL-encoded query string and returns
// a map listing the values specified for each key.
// ParseQuery always returns a non-nil map containing all the
// valid query parameters found; err describes the first decoding error
// encountered, if any.
func ParseQuery(query string) (m Values, err os.Error) {
	m = make(Values)
	err = parseQuery(m, query)
	return
}

func parseQuery(m Values, query string) (err os.Error) {
	for query != "" {
		key := query
		if i := strings.IndexAny(key, "&;"); i >= 0 {
			key, query = key[:i], key[i+1:]
		} else {
			query = ""
		}
		if key == "" {
			continue
		}
		value := ""
		if i := strings.Index(key, "="); i >= 0 {
			key, value = key[:i], key[i+1:]
		}
		key, err1 := QueryUnescape(key)
		if err1 != nil {
			err = err1
			continue
		}
		value, err1 = QueryUnescape(value)
		if err1 != nil {
			err = err1
			continue
		}
		m[key] = append(m[key], value)
	}
	return err
}

// Encode encodes the values into ``URL encoded'' form.
// e.g. "foo=bar&bar=baz"
func (v Values) Encode() string {
	if v == nil {
		return ""
	}
	parts := make([]string, 0, len(v)) // will be large enough for most uses
	for k, vs := range v {
		prefix := QueryEscape(k) + "="
		for _, v := range vs {
			parts = append(parts, prefix+QueryEscape(v))
		}
	}
	return strings.Join(parts, "&")
}

// resolvePath applies special path segments from refs and applies
// them to base, per RFC 2396.
func resolvePath(basepath string, refpath string) string {
	base := strings.Split(basepath, "/")
	refs := strings.Split(refpath, "/")
	if len(base) == 0 {
		base = []string{""}
	}
	for idx, ref := range refs {
		switch {
		case ref == ".":
			base[len(base)-1] = ""
		case ref == "..":
			newLen := len(base) - 1
			if newLen < 1 {
				newLen = 1
			}
			base = base[0:newLen]
			base[len(base)-1] = ""
		default:
			if idx == 0 || base[len(base)-1] == "" {
				base[len(base)-1] = ref
			} else {
				base = append(base, ref)
			}
		}
	}
	return strings.Join(base, "/")
}

// IsAbs returns true if the URL is absolute.
func (url *URL) IsAbs() bool {
	return url.Scheme != ""
}

// Parse parses a URL in the context of a base URL.  The URL in ref
// may be relative or absolute.  Parse returns nil, err on parse
// failure, otherwise its return value is the same as ResolveReference.
func (base *URL) Parse(ref string) (*URL, os.Error) {
	refurl, err := Parse(ref)
	if err != nil {
		return nil, err
	}
	return base.ResolveReference(refurl), nil
}

// ResolveReference resolves a URI reference to an absolute URI from
// an absolute base URI, per RFC 2396 Section 5.2.  The URI reference
// may be relative or absolute.  ResolveReference always returns a new
// URL instance, even if the returned URL is identical to either the
// base or reference. If ref is an absolute URL, then ResolveReference
// ignores base and returns a copy of ref.
func (base *URL) ResolveReference(ref *URL) *URL {
	url := new(URL)
	switch {
	case ref.IsAbs():
		*url = *ref
	default:
		// relativeURI   = ( net_path | abs_path | rel_path ) [ "?" query ]
		*url = *base
		if ref.RawAuthority != "" {
			// The "net_path" case.
			url.RawAuthority = ref.RawAuthority
			url.Host = ref.Host
			url.RawUserinfo = ref.RawUserinfo
		}
		switch {
		case url.OpaquePath:
			url.Path = ref.Path
			url.RawPath = ref.RawPath
			url.RawQuery = ref.RawQuery
		case strings.HasPrefix(ref.Path, "/"):
			// The "abs_path" case.
			url.Path = ref.Path
			url.RawPath = ref.RawPath
			url.RawQuery = ref.RawQuery
		default:
			// The "rel_path" case.
			path := resolvePath(base.Path, ref.Path)
			if !strings.HasPrefix(path, "/") {
				path = "/" + path
			}
			url.Path = path
			url.RawPath = url.Path
			url.RawQuery = ref.RawQuery
			if ref.RawQuery != "" {
				url.RawPath += "?" + url.RawQuery
			}
		}

		url.Fragment = ref.Fragment
	}
	url.Raw = url.String()
	return url
}

// Query parses RawQuery and returns the corresponding values.
func (u *URL) Query() Values {
	v, _ := ParseQuery(u.RawQuery)
	return v
}

// EncodedPath returns the URL's path in "URL path encoded" form.
func (u *URL) EncodedPath() string {
	return escape(u.Path, encodePath)
}
