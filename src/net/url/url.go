// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package url parses URLs and implements query escaping.
package url

// See RFC 3986. This package generally follows RFC 3986, except where
// it deviates for compatibility reasons. When sending changes, first
// search old issues for history on decisions. Unit tests should also
// contain references to issue numbers with details.

import (
	"errors"
	"fmt"
	"maps"
	"path"
	"slices"
	"strconv"
	"strings"
	_ "unsafe" // for linkname
)

// Error reports an error and the operation and URL that caused it.
type Error struct {
	Op  string
	URL string
	Err error
}

func (e *Error) Unwrap() error { return e.Err }
func (e *Error) Error() string { return fmt.Sprintf("%s %q: %s", e.Op, e.URL, e.Err) }

func (e *Error) Timeout() bool {
	t, ok := e.Err.(interface {
		Timeout() bool
	})
	return ok && t.Timeout()
}

func (e *Error) Temporary() bool {
	t, ok := e.Err.(interface {
		Temporary() bool
	})
	return ok && t.Temporary()
}

const upperhex = "0123456789ABCDEF"

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
	default:
		panic("invalid hex character")
	}
}

type encoding int

const (
	encodePath encoding = 1 + iota
	encodePathSegment
	encodeHost
	encodeZone
	encodeUserPassword
	encodeQueryComponent
	encodeFragment
)

type EscapeError string

func (e EscapeError) Error() string {
	return "invalid URL escape " + strconv.Quote(string(e))
}

type InvalidHostError string

func (e InvalidHostError) Error() string {
	return "invalid character " + strconv.Quote(string(e)) + " in host name"
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

// QueryUnescape does the inverse transformation of [QueryEscape],
// converting each 3-byte encoded substring of the form "%AB" into the
// hex-decoded byte 0xAB.
// It returns an error if any % is not followed by two hexadecimal
// digits.
func QueryUnescape(s string) (string, error) {
	return unescape(s, encodeQueryComponent)
}

// PathUnescape does the inverse transformation of [PathEscape],
// converting each 3-byte encoded substring of the form "%AB" into the
// hex-decoded byte 0xAB. It returns an error if any % is not followed
// by two hexadecimal digits.
//
// PathUnescape is identical to [QueryUnescape] except that it does not
// unescape '+' to ' ' (space).
func PathUnescape(s string) (string, error) {
	return unescape(s, encodePathSegment)
}

// unescape unescapes a string; the mode specifies
// which section of the URL string is being unescaped.
func unescape(s string, mode encoding) (string, error) {
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
					s = s[:3]
				}
				return "", EscapeError(s)
			}
			// Per https://tools.ietf.org/html/rfc3986#page-21
			// in the host component %-encoding can only be used
			// for non-ASCII bytes.
			// But https://tools.ietf.org/html/rfc6874#section-2
			// introduces %25 being allowed to escape a percent sign
			// in IPv6 scoped-address literals. Yay.
			if mode == encodeHost && unhex(s[i+1]) < 8 && s[i:i+3] != "%25" {
				return "", EscapeError(s[i : i+3])
			}
			if mode == encodeZone {
				// RFC 6874 says basically "anything goes" for zone identifiers
				// and that even non-ASCII can be redundantly escaped,
				// but it seems prudent to restrict %-escaped bytes here to those
				// that are valid host name bytes in their unescaped form.
				// That is, you can use escaping in the zone identifier but not
				// to introduce bytes you couldn't just write directly.
				// But Windows puts spaces here! Yay.
				v := unhex(s[i+1])<<4 | unhex(s[i+2])
				if s[i:i+3] != "%25" && v != ' ' && shouldEscape(v, encodeHost) {
					return "", EscapeError(s[i : i+3])
				}
			}
			i += 3
		case '+':
			hasPlus = mode == encodeQueryComponent
			i++
		default:
			if (mode == encodeHost || mode == encodeZone) && s[i] < 0x80 && shouldEscape(s[i], mode) {
				return "", InvalidHostError(s[i : i+1])
			}
			i++
		}
	}

	if n == 0 && !hasPlus {
		return s, nil
	}

	var t strings.Builder
	t.Grow(len(s) - 2*n)
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '%':
			t.WriteByte(unhex(s[i+1])<<4 | unhex(s[i+2]))
			i += 2
		case '+':
			if mode == encodeQueryComponent {
				t.WriteByte(' ')
			} else {
				t.WriteByte('+')
			}
		default:
			t.WriteByte(s[i])
		}
	}
	return t.String(), nil
}

// QueryEscape escapes the string so it can be safely placed
// inside a [URL] query.
func QueryEscape(s string) string {
	return escape(s, encodeQueryComponent)
}

// PathEscape escapes the string so it can be safely placed inside a [URL] path segment,
// replacing special characters (including /) with %XX sequences as needed.
func PathEscape(s string) string {
	return escape(s, encodePathSegment)
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

	var buf [64]byte
	var t []byte

	required := len(s) + 2*hexCount
	if required <= len(buf) {
		t = buf[:required]
	} else {
		t = make([]byte, required)
	}

	if hexCount == 0 {
		copy(t, s)
		for i := 0; i < len(s); i++ {
			if s[i] == ' ' {
				t[i] = '+'
			}
		}
		return string(t)
	}

	j := 0
	for i := 0; i < len(s); i++ {
		switch c := s[i]; {
		case c == ' ' && mode == encodeQueryComponent:
			t[j] = '+'
			j++
		case shouldEscape(c, mode):
			t[j] = '%'
			t[j+1] = upperhex[c>>4]
			t[j+2] = upperhex[c&15]
			j += 3
		default:
			t[j] = s[i]
			j++
		}
	}
	return string(t)
}

// A URL represents a parsed URL (technically, a URI reference).
//
// The general form represented is:
//
//	[scheme:][//[userinfo@]host][/]path[?query][#fragment]
//
// URLs that do not start with a slash after the scheme are interpreted as:
//
//	scheme:opaque[?query][#fragment]
//
// The Host field contains the host and port subcomponents of the URL.
// When the port is present, it is separated from the host with a colon.
// When the host is an IPv6 address, it must be enclosed in square brackets:
// "[fe80::1]:80". The [net.JoinHostPort] function combines a host and port
// into a string suitable for the Host field, adding square brackets to
// the host when necessary.
//
// Note that the Path field is stored in decoded form: /%47%6f%2f becomes /Go/.
// A consequence is that it is impossible to tell which slashes in the Path were
// slashes in the raw URL and which were %2f. This distinction is rarely important,
// but when it is, the code should use the [URL.EscapedPath] method, which preserves
// the original encoding of Path.
//
// The RawPath field is an optional field which is only set when the default
// encoding of Path is different from the escaped path. See the EscapedPath method
// for more details.
//
// URL's String method uses the EscapedPath method to obtain the path.
type URL struct {
	Scheme      string
	Opaque      string    // encoded opaque data
	User        *Userinfo // username and password information
	Host        string    // host or host:port (see Hostname and Port methods)
	Path        string    // path (relative paths may omit leading slash)
	RawPath     string    // encoded path hint (see EscapedPath method)
	OmitHost    bool      // do not emit empty host (authority)
	ForceQuery  bool      // append a query ('?') even if RawQuery is empty
	RawQuery    string    // encoded query values, without '?'
	Fragment    string    // fragment for references, without '#'
	RawFragment string    // encoded fragment hint (see EscapedFragment method)
}

// User returns a [Userinfo] containing the provided username
// and no password set.
func User(username string) *Userinfo {
	return &Userinfo{username, "", false}
}

// UserPassword returns a [Userinfo] containing the provided username
// and password.
//
// This functionality should only be used with legacy web sites.
// RFC 2396 warns that interpreting Userinfo this way
// “is NOT RECOMMENDED, because the passing of authentication
// information in clear text (such as URI) has proven to be a
// security risk in almost every case where it has been used.”
func UserPassword(username, password string) *Userinfo {
	return &Userinfo{username, password, true}
}

// The Userinfo type is an immutable encapsulation of username and
// password details for a [URL]. An existing Userinfo value is guaranteed
// to have a username set (potentially empty, as allowed by RFC 2396),
// and optionally a password.
type Userinfo struct {
	username    string
	password    string
	passwordSet bool
}

// Username returns the username.
func (u *Userinfo) Username() string {
	if u == nil {
		return ""
	}
	return u.username
}

// Password returns the password in case it is set, and whether it is set.
func (u *Userinfo) Password() (string, bool) {
	if u == nil {
		return "", false
	}
	return u.password, u.passwordSet
}

// String returns the encoded userinfo information in the standard form
// of "username[:password]".
func (u *Userinfo) String() string {
	if u == nil {
		return ""
	}
	s := escape(u.username, encodeUserPassword)
	if u.passwordSet {
		s += ":" + escape(u.password, encodeUserPassword)
	}
	return s
}

// Maybe rawURL is of the form scheme:path.
// (Scheme must be [a-zA-Z][a-zA-Z0-9+.-]*)
// If so, return scheme, path; else return "", rawURL.
func getScheme(rawURL string) (scheme, path string, err error) {
	for i := 0; i < len(rawURL); i++ {
		c := rawURL[i]
		switch {
		case 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z':
		// do nothing
		case '0' <= c && c <= '9' || c == '+' || c == '-' || c == '.':
			if i == 0 {
				return "", rawURL, nil
			}
		case c == ':':
			if i == 0 {
				return "", "", errors.New("missing protocol scheme")
			}
			return rawURL[:i], rawURL[i+1:], nil
		default:
			// we have encountered an invalid character,
			// so there is no valid scheme
			return "", rawURL, nil
		}
	}
	return "", rawURL, nil
}

// Parse parses a raw url into a [URL] structure.
//
// The url may be relative (a path, without a host) or absolute
// (starting with a scheme). Trying to parse a hostname and path
// without a scheme is invalid but may not necessarily return an
// error, due to parsing ambiguities.
func Parse(rawURL string) (*URL, error) {
	// Cut off #frag
	u, frag, _ := strings.Cut(rawURL, "#")
	url, err := parse(u, false)
	if err != nil {
		return nil, &Error{"parse", u, err}
	}
	if frag == "" {
		return url, nil
	}
	if err = url.setFragment(frag); err != nil {
		return nil, &Error{"parse", rawURL, err}
	}
	return url, nil
}

// ParseRequestURI parses a raw url into a [URL] structure. It assumes that
// url was received in an HTTP request, so the url is interpreted
// only as an absolute URI or an absolute path.
// The string url is assumed not to have a #fragment suffix.
// (Web browsers strip #fragment before sending the URL to a web server.)
func ParseRequestURI(rawURL string) (*URL, error) {
	url, err := parse(rawURL, true)
	if err != nil {
		return nil, &Error{"parse", rawURL, err}
	}
	return url, nil
}

// parse parses a URL from a string in one of two contexts. If
// viaRequest is true, the URL is assumed to have arrived via an HTTP request,
// in which case only absolute URLs or path-absolute relative URLs are allowed.
// If viaRequest is false, all forms of relative URLs are allowed.
func parse(rawURL string, viaRequest bool) (*URL, error) {
	var rest string
	var err error

	if stringContainsCTLByte(rawURL) {
		return nil, errors.New("net/url: invalid control character in URL")
	}

	if rawURL == "" && viaRequest {
		return nil, errors.New("empty url")
	}
	url := new(URL)

	if rawURL == "*" {
		url.Path = "*"
		return url, nil
	}

	// Split off possible leading "http:", "mailto:", etc.
	// Cannot contain escaped characters.
	if url.Scheme, rest, err = getScheme(rawURL); err != nil {
		return nil, err
	}
	url.Scheme = strings.ToLower(url.Scheme)

	if strings.HasSuffix(rest, "?") && strings.Count(rest, "?") == 1 {
		url.ForceQuery = true
		rest = rest[:len(rest)-1]
	} else {
		rest, url.RawQuery, _ = strings.Cut(rest, "?")
	}

	if !strings.HasPrefix(rest, "/") {
		if url.Scheme != "" {
			// We consider rootless paths per RFC 3986 as opaque.
			url.Opaque = rest
			return url, nil
		}
		if viaRequest {
			return nil, errors.New("invalid URI for request")
		}

		// Avoid confusion with malformed schemes, like cache_object:foo/bar.
		// See golang.org/issue/16822.
		//
		// RFC 3986, §3.3:
		// In addition, a URI reference (Section 4.1) may be a relative-path reference,
		// in which case the first path segment cannot contain a colon (":") character.
		if segment, _, _ := strings.Cut(rest, "/"); strings.Contains(segment, ":") {
			// First path segment has colon. Not allowed in relative URL.
			return nil, errors.New("first path segment in URL cannot contain colon")
		}
	}

	if (url.Scheme != "" || !viaRequest && !strings.HasPrefix(rest, "///")) && strings.HasPrefix(rest, "//") {
		var authority string
		authority, rest = rest[2:], ""
		if i := strings.Index(authority, "/"); i >= 0 {
			authority, rest = authority[:i], authority[i:]
		}
		url.User, url.Host, err = parseAuthority(authority)
		if err != nil {
			return nil, err
		}
	} else if url.Scheme != "" && strings.HasPrefix(rest, "/") {
		// OmitHost is set to true when rawURL has an empty host (authority).
		// See golang.org/issue/46059.
		url.OmitHost = true
	}

	// Set Path and, optionally, RawPath.
	// RawPath is a hint of the encoding of Path. We don't want to set it if
	// the default escaping of Path is equivalent, to help make sure that people
	// don't rely on it in general.
	if err := url.setPath(rest); err != nil {
		return nil, err
	}
	return url, nil
}

func parseAuthority(authority string) (user *Userinfo, host string, err error) {
	i := strings.LastIndex(authority, "@")
	if i < 0 {
		host, err = parseHost(authority)
	} else {
		host, err = parseHost(authority[i+1:])
	}
	if err != nil {
		return nil, "", err
	}
	if i < 0 {
		return nil, host, nil
	}
	userinfo := authority[:i]
	if !validUserinfo(userinfo) {
		return nil, "", errors.New("net/url: invalid userinfo")
	}
	if !strings.Contains(userinfo, ":") {
		if userinfo, err = unescape(userinfo, encodeUserPassword); err != nil {
			return nil, "", err
		}
		user = User(userinfo)
	} else {
		username, password, _ := strings.Cut(userinfo, ":")
		if username, err = unescape(username, encodeUserPassword); err != nil {
			return nil, "", err
		}
		if password, err = unescape(password, encodeUserPassword); err != nil {
			return nil, "", err
		}
		user = UserPassword(username, password)
	}
	return user, host, nil
}

// parseHost parses host as an authority without user
// information. That is, as host[:port].
func parseHost(host string) (string, error) {
	if strings.HasPrefix(host, "[") {
		// Parse an IP-Literal in RFC 3986 and RFC 6874.
		// E.g., "[fe80::1]", "[fe80::1%25en0]", "[fe80::1]:80".
		i := strings.LastIndex(host, "]")
		if i < 0 {
			return "", errors.New("missing ']' in host")
		}
		colonPort := host[i+1:]
		if !validOptionalPort(colonPort) {
			return "", fmt.Errorf("invalid port %q after host", colonPort)
		}

		// RFC 6874 defines that %25 (%-encoded percent) introduces
		// the zone identifier, and the zone identifier can use basically
		// any %-encoding it likes. That's different from the host, which
		// can only %-encode non-ASCII bytes.
		// We do impose some restrictions on the zone, to avoid stupidity
		// like newlines.
		zone := strings.Index(host[:i], "%25")
		if zone >= 0 {
			host1, err := unescape(host[:zone], encodeHost)
			if err != nil {
				return "", err
			}
			host2, err := unescape(host[zone:i], encodeZone)
			if err != nil {
				return "", err
			}
			host3, err := unescape(host[i:], encodeHost)
			if err != nil {
				return "", err
			}
			return host1 + host2 + host3, nil
		}
	} else if i := strings.LastIndex(host, ":"); i != -1 {
		colonPort := host[i:]
		if !validOptionalPort(colonPort) {
			return "", fmt.Errorf("invalid port %q after host", colonPort)
		}
	}

	var err error
	if host, err = unescape(host, encodeHost); err != nil {
		return "", err
	}
	return host, nil
}

// setPath sets the Path and RawPath fields of the URL based on the provided
// escaped path p. It maintains the invariant that RawPath is only specified
// when it differs from the default encoding of the path.
// For example:
// - setPath("/foo/bar")   will set Path="/foo/bar" and RawPath=""
// - setPath("/foo%2fbar") will set Path="/foo/bar" and RawPath="/foo%2fbar"
// setPath will return an error only if the provided path contains an invalid
// escaping.
//
// setPath should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/sagernet/sing
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname badSetPath net/url.(*URL).setPath
func (u *URL) setPath(p string) error {
	path, err := unescape(p, encodePath)
	if err != nil {
		return err
	}
	u.Path = path
	if escp := escape(path, encodePath); p == escp {
		// Default encoding is fine.
		u.RawPath = ""
	} else {
		u.RawPath = p
	}
	return nil
}

// for linkname because we cannot linkname methods directly
func badSetPath(*URL, string) error

// EscapedPath returns the escaped form of u.Path.
// In general there are multiple possible escaped forms of any path.
// EscapedPath returns u.RawPath when it is a valid escaping of u.Path.
// Otherwise EscapedPath ignores u.RawPath and computes an escaped
// form on its own.
// The [URL.String] and [URL.RequestURI] methods use EscapedPath to construct
// their results.
// In general, code should call EscapedPath instead of
// reading u.RawPath directly.
func (u *URL) EscapedPath() string {
	if u.RawPath != "" && validEncoded(u.RawPath, encodePath) {
		p, err := unescape(u.RawPath, encodePath)
		if err == nil && p == u.Path {
			return u.RawPath
		}
	}
	if u.Path == "*" {
		return "*" // don't escape (Issue 11202)
	}
	return escape(u.Path, encodePath)
}

// validEncoded reports whether s is a valid encoded path or fragment,
// according to mode.
// It must not contain any bytes that require escaping during encoding.
func validEncoded(s string, mode encoding) bool {
	for i := 0; i < len(s); i++ {
		// RFC 3986, Appendix A.
		// pchar = unreserved / pct-encoded / sub-delims / ":" / "@".
		// shouldEscape is not quite compliant with the RFC,
		// so we check the sub-delims ourselves and let
		// shouldEscape handle the others.
		switch s[i] {
		case '!', '$', '&', '\'', '(', ')', '*', '+', ',', ';', '=', ':', '@':
			// ok
		case '[', ']':
			// ok - not specified in RFC 3986 but left alone by modern browsers
		case '%':
			// ok - percent encoded, will decode
		default:
			if shouldEscape(s[i], mode) {
				return false
			}
		}
	}
	return true
}

// setFragment is like setPath but for Fragment/RawFragment.
func (u *URL) setFragment(f string) error {
	frag, err := unescape(f, encodeFragment)
	if err != nil {
		return err
	}
	u.Fragment = frag
	if escf := escape(frag, encodeFragment); f == escf {
		// Default encoding is fine.
		u.RawFragment = ""
	} else {
		u.RawFragment = f
	}
	return nil
}

// EscapedFragment returns the escaped form of u.Fragment.
// In general there are multiple possible escaped forms of any fragment.
// EscapedFragment returns u.RawFragment when it is a valid escaping of u.Fragment.
// Otherwise EscapedFragment ignores u.RawFragment and computes an escaped
// form on its own.
// The [URL.String] method uses EscapedFragment to construct its result.
// In general, code should call EscapedFragment instead of
// reading u.RawFragment directly.
func (u *URL) EscapedFragment() string {
	if u.RawFragment != "" && validEncoded(u.RawFragment, encodeFragment) {
		f, err := unescape(u.RawFragment, encodeFragment)
		if err == nil && f == u.Fragment {
			return u.RawFragment
		}
	}
	return escape(u.Fragment, encodeFragment)
}

// validOptionalPort reports whether port is either an empty string
// or matches /^:\d*$/
func validOptionalPort(port string) bool {
	if port == "" {
		return true
	}
	if port[0] != ':' {
		return false
	}
	for _, b := range port[1:] {
		if b < '0' || b > '9' {
			return false
		}
	}
	return true
}

// String reassembles the [URL] into a valid URL string.
// The general form of the result is one of:
//
//	scheme:opaque?query#fragment
//	scheme://userinfo@host/path?query#fragment
//
// If u.Opaque is non-empty, String uses the first form;
// otherwise it uses the second form.
// Any non-ASCII characters in host are escaped.
// To obtain the path, String uses u.EscapedPath().
//
// In the second form, the following rules apply:
//   - if u.Scheme is empty, scheme: is omitted.
//   - if u.User is nil, userinfo@ is omitted.
//   - if u.Host is empty, host/ is omitted.
//   - if u.Scheme and u.Host are empty and u.User is nil,
//     the entire scheme://userinfo@host/ is omitted.
//   - if u.Host is non-empty and u.Path begins with a /,
//     the form host/path does not add its own /.
//   - if u.RawQuery is empty, ?query is omitted.
//   - if u.Fragment is empty, #fragment is omitted.
func (u *URL) String() string {
	var buf strings.Builder

	n := len(u.Scheme)
	if u.Opaque != "" {
		n += len(u.Opaque)
	} else {
		if !u.OmitHost && (u.Scheme != "" || u.Host != "" || u.User != nil) {
			username := u.User.Username()
			password, _ := u.User.Password()
			n += len(username) + len(password) + len(u.Host)
		}
		n += len(u.Path)
	}
	n += len(u.RawQuery) + len(u.RawFragment)
	n += len(":" + "//" + "//" + ":" + "@" + "/" + "./" + "?" + "#")
	buf.Grow(n)

	if u.Scheme != "" {
		buf.WriteString(u.Scheme)
		buf.WriteByte(':')
	}
	if u.Opaque != "" {
		buf.WriteString(u.Opaque)
	} else {
		if u.Scheme != "" || u.Host != "" || u.User != nil {
			if u.OmitHost && u.Host == "" && u.User == nil {
				// omit empty host
			} else {
				if u.Host != "" || u.Path != "" || u.User != nil {
					buf.WriteString("//")
				}
				if ui := u.User; ui != nil {
					buf.WriteString(ui.String())
					buf.WriteByte('@')
				}
				if h := u.Host; h != "" {
					buf.WriteString(escape(h, encodeHost))
				}
			}
		}
		path := u.EscapedPath()
		if path != "" && path[0] != '/' && u.Host != "" {
			buf.WriteByte('/')
		}
		if buf.Len() == 0 {
			// RFC 3986 §4.2
			// A path segment that contains a colon character (e.g., "this:that")
			// cannot be used as the first segment of a relative-path reference, as
			// it would be mistaken for a scheme name. Such a segment must be
			// preceded by a dot-segment (e.g., "./this:that") to make a relative-
			// path reference.
			if segment, _, _ := strings.Cut(path, "/"); strings.Contains(segment, ":") {
				buf.WriteString("./")
			}
		}
		buf.WriteString(path)
	}
	if u.ForceQuery || u.RawQuery != "" {
		buf.WriteByte('?')
		buf.WriteString(u.RawQuery)
	}
	if u.Fragment != "" {
		buf.WriteByte('#')
		buf.WriteString(u.EscapedFragment())
	}
	return buf.String()
}

// Redacted is like [URL.String] but replaces any password with "xxxxx".
// Only the password in u.User is redacted.
func (u *URL) Redacted() string {
	if u == nil {
		return ""
	}

	ru := *u
	if _, has := ru.User.Password(); has {
		ru.User = UserPassword(ru.User.Username(), "xxxxx")
	}
	return ru.String()
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
	vs := v[key]
	if len(vs) == 0 {
		return ""
	}
	return vs[0]
}

// Set sets the key to value. It replaces any existing
// values.
func (v Values) Set(key, value string) {
	v[key] = []string{value}
}

// Add adds the value to key. It appends to any existing
// values associated with key.
func (v Values) Add(key, value string) {
	v[key] = append(v[key], value)
}

// Del deletes the values associated with key.
func (v Values) Del(key string) {
	delete(v, key)
}

// Has checks whether a given key is set.
func (v Values) Has(key string) bool {
	_, ok := v[key]
	return ok
}

// ParseQuery parses the URL-encoded query string and returns
// a map listing the values specified for each key.
// ParseQuery always returns a non-nil map containing all the
// valid query parameters found; err describes the first decoding error
// encountered, if any.
//
// Query is expected to be a list of key=value settings separated by ampersands.
// A setting without an equals sign is interpreted as a key set to an empty
// value.
// Settings containing a non-URL-encoded semicolon are considered invalid.
func ParseQuery(query string) (Values, error) {
	m := make(Values)
	err := parseQuery(m, query)
	return m, err
}

func parseQuery(m Values, query string) (err error) {
	for query != "" {
		var key string
		key, query, _ = strings.Cut(query, "&")
		if strings.Contains(key, ";") {
			err = fmt.Errorf("invalid semicolon separator in query")
			continue
		}
		if key == "" {
			continue
		}
		key, value, _ := strings.Cut(key, "=")
		key, err1 := QueryUnescape(key)
		if err1 != nil {
			if err == nil {
				err = err1
			}
			continue
		}
		value, err1 = QueryUnescape(value)
		if err1 != nil {
			if err == nil {
				err = err1
			}
			continue
		}
		m[key] = append(m[key], value)
	}
	return err
}

// Encode encodes the values into “URL encoded” form
// ("bar=baz&foo=quux") sorted by key.
func (v Values) Encode() string {
	if len(v) == 0 {
		return ""
	}
	var buf strings.Builder
	for _, k := range slices.Sorted(maps.Keys(v)) {
		vs := v[k]
		keyEscaped := QueryEscape(k)
		for _, v := range vs {
			if buf.Len() > 0 {
				buf.WriteByte('&')
			}
			buf.WriteString(keyEscaped)
			buf.WriteByte('=')
			buf.WriteString(QueryEscape(v))
		}
	}
	return buf.String()
}

// resolvePath applies special path segments from refs and applies
// them to base, per RFC 3986.
func resolvePath(base, ref string) string {
	var full string
	if ref == "" {
		full = base
	} else if ref[0] != '/' {
		i := strings.LastIndex(base, "/")
		full = base[:i+1] + ref
	} else {
		full = ref
	}
	if full == "" {
		return ""
	}

	var (
		elem string
		dst  strings.Builder
	)
	first := true
	remaining := full
	// We want to return a leading '/', so write it now.
	dst.WriteByte('/')
	found := true
	for found {
		elem, remaining, found = strings.Cut(remaining, "/")
		if elem == "." {
			first = false
			// drop
			continue
		}

		if elem == ".." {
			// Ignore the leading '/' we already wrote.
			str := dst.String()[1:]
			index := strings.LastIndexByte(str, '/')

			dst.Reset()
			dst.WriteByte('/')
			if index == -1 {
				first = true
			} else {
				dst.WriteString(str[:index])
			}
		} else {
			if !first {
				dst.WriteByte('/')
			}
			dst.WriteString(elem)
			first = false
		}
	}

	if elem == "." || elem == ".." {
		dst.WriteByte('/')
	}

	// We wrote an initial '/', but we don't want two.
	r := dst.String()
	if len(r) > 1 && r[1] == '/' {
		r = r[1:]
	}
	return r
}

// IsAbs reports whether the [URL] is absolute.
// Absolute means that it has a non-empty scheme.
func (u *URL) IsAbs() bool {
	return u.Scheme != ""
}

// Parse parses a [URL] in the context of the receiver. The provided URL
// may be relative or absolute. Parse returns nil, err on parse
// failure, otherwise its return value is the same as [URL.ResolveReference].
func (u *URL) Parse(ref string) (*URL, error) {
	refURL, err := Parse(ref)
	if err != nil {
		return nil, err
	}
	return u.ResolveReference(refURL), nil
}

// ResolveReference resolves a URI reference to an absolute URI from
// an absolute base URI u, per RFC 3986 Section 5.2. The URI reference
// may be relative or absolute. ResolveReference always returns a new
// [URL] instance, even if the returned URL is identical to either the
// base or reference. If ref is an absolute URL, then ResolveReference
// ignores base and returns a copy of ref.
func (u *URL) ResolveReference(ref *URL) *URL {
	url := *ref
	if ref.Scheme == "" {
		url.Scheme = u.Scheme
	}
	if ref.Scheme != "" || ref.Host != "" || ref.User != nil {
		// The "absoluteURI" or "net_path" cases.
		// We can ignore the error from setPath since we know we provided a
		// validly-escaped path.
		url.setPath(resolvePath(ref.EscapedPath(), ""))
		return &url
	}
	if ref.Opaque != "" {
		url.User = nil
		url.Host = ""
		url.Path = ""
		return &url
	}
	if ref.Path == "" && !ref.ForceQuery && ref.RawQuery == "" {
		url.RawQuery = u.RawQuery
		if ref.Fragment == "" {
			url.Fragment = u.Fragment
			url.RawFragment = u.RawFragment
		}
	}
	if ref.Path == "" && u.Opaque != "" {
		url.Opaque = u.Opaque
		url.User = nil
		url.Host = ""
		url.Path = ""
		return &url
	}
	// The "abs_path" or "rel_path" cases.
	url.Host = u.Host
	url.User = u.User
	url.setPath(resolvePath(u.EscapedPath(), ref.EscapedPath()))
	return &url
}

// Query parses RawQuery and returns the corresponding values.
// It silently discards malformed value pairs.
// To check errors use [ParseQuery].
func (u *URL) Query() Values {
	v, _ := ParseQuery(u.RawQuery)
	return v
}

// RequestURI returns the encoded path?query or opaque?query
// string that would be used in an HTTP request for u.
func (u *URL) RequestURI() string {
	result := u.Opaque
	if result == "" {
		result = u.EscapedPath()
		if result == "" {
			result = "/"
		}
	} else {
		if strings.HasPrefix(result, "//") {
			result = u.Scheme + ":" + result
		}
	}
	if u.ForceQuery || u.RawQuery != "" {
		result += "?" + u.RawQuery
	}
	return result
}

// Hostname returns u.Host, stripping any valid port number if present.
//
// If the result is enclosed in square brackets, as literal IPv6 addresses are,
// the square brackets are removed from the result.
func (u *URL) Hostname() string {
	host, _ := splitHostPort(u.Host)
	return host
}

// Port returns the port part of u.Host, without the leading colon.
//
// If u.Host doesn't contain a valid numeric port, Port returns an empty string.
func (u *URL) Port() string {
	_, port := splitHostPort(u.Host)
	return port
}

// splitHostPort separates host and port. If the port is not valid, it returns
// the entire input as host, and it doesn't check the validity of the host.
// Unlike net.SplitHostPort, but per RFC 3986, it requires ports to be numeric.
func splitHostPort(hostPort string) (host, port string) {
	host = hostPort

	colon := strings.LastIndexByte(host, ':')
	if colon != -1 && validOptionalPort(host[colon:]) {
		host, port = host[:colon], host[colon+1:]
	}

	if strings.HasPrefix(host, "[") && strings.HasSuffix(host, "]") {
		host = host[1 : len(host)-1]
	}

	return
}

// Marshaling interface implementations.
// Would like to implement MarshalText/UnmarshalText but that will change the JSON representation of URLs.

func (u *URL) MarshalBinary() (text []byte, err error) {
	return u.AppendBinary(nil)
}

func (u *URL) AppendBinary(b []byte) ([]byte, error) {
	return append(b, u.String()...), nil
}

func (u *URL) UnmarshalBinary(text []byte) error {
	u1, err := Parse(string(text))
	if err != nil {
		return err
	}
	*u = *u1
	return nil
}

// JoinPath returns a new [URL] with the provided path elements joined to
// any existing path and the resulting path cleaned of any ./ or ../ elements.
// Any sequences of multiple / characters will be reduced to a single /.
func (u *URL) JoinPath(elem ...string) *URL {
	elem = append([]string{u.EscapedPath()}, elem...)
	var p string
	if !strings.HasPrefix(elem[0], "/") {
		// Return a relative path if u is relative,
		// but ensure that it contains no ../ elements.
		elem[0] = "/" + elem[0]
		p = path.Join(elem...)[1:]
	} else {
		p = path.Join(elem...)
	}
	// path.Join will remove any trailing slashes.
	// Preserve at least one.
	if strings.HasSuffix(elem[len(elem)-1], "/") && !strings.HasSuffix(p, "/") {
		p += "/"
	}
	url := *u
	url.setPath(p)
	return &url
}

// validUserinfo reports whether s is a valid userinfo string per RFC 3986
// Section 3.2.1:
//
//	userinfo    = *( unreserved / pct-encoded / sub-delims / ":" )
//	unreserved  = ALPHA / DIGIT / "-" / "." / "_" / "~"
//	sub-delims  = "!" / "$" / "&" / "'" / "(" / ")"
//	              / "*" / "+" / "," / ";" / "="
//
// It doesn't validate pct-encoded. The caller does that via func unescape.
func validUserinfo(s string) bool {
	for _, r := range s {
		if 'A' <= r && r <= 'Z' {
			continue
		}
		if 'a' <= r && r <= 'z' {
			continue
		}
		if '0' <= r && r <= '9' {
			continue
		}
		switch r {
		case '-', '.', '_', ':', '~', '!', '$', '&', '\'',
			'(', ')', '*', '+', ',', ';', '=', '%', '@':
			continue
		default:
			return false
		}
	}
	return true
}

// stringContainsCTLByte reports whether s contains any ASCII control character.
func stringContainsCTLByte(s string) bool {
	for i := 0; i < len(s); i++ {
		b := s[i]
		if b < ' ' || b == 0x7f {
			return true
		}
	}
	return false
}

// JoinPath returns a [URL] string with the provided path elements joined to
// the existing path of base and the resulting path cleaned of any ./ or ../ elements.
func JoinPath(base string, elem ...string) (result string, err error) {
	url, err := Parse(base)
	if err != nil {
		return
	}
	result = url.JoinPath(elem...).String()
	return
}
