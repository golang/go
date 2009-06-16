// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parse URLs (actually URIs, but that seems overly pedantic).
// RFC 2396

package http

import (
	"os";
	"strings"
)

// Errors introduced by ParseURL.
type BadURL struct {
	os.ErrorString
}

func ishex(c byte) bool {
	switch {
	case '0' <= c && c <= '9':
		return true;
	case 'a' <= c && c <= 'f':
		return true;
	case 'A' <= c && c <= 'F':
		return true;
	}
	return false
}

func unhex(c byte) byte {
	switch {
	case '0' <= c && c <= '9':
		return c - '0';
	case 'a' <= c && c <= 'f':
		return c - 'a' + 10;
	case 'A' <= c && c <= 'F':
		return c - 'A' + 10;
	}
	return 0
}

// Return true if the specified character should be escaped when appearing in a
// URL string.
//
// TODO: for now, this is a hack; it only flags a few common characters that have
// special meaning in URLs.  That will get the job done in the common cases.
func shouldEscape(c byte) bool {
	switch c {
	case ' ', '?', '&', '=', '#', '+', '%':
		return true;
	}
	return false;
}

// URLUnescape unescapes a URL-encoded string,
// converting %AB into the byte 0xAB and '+' into ' ' (space).
// It returns a BadURL error if any % is not followed
// by two hexadecimal digits.
func URLUnescape(s string) (string, os.Error) {
	// Count %, check that they're well-formed.
	n := 0;
	anyPlusses := false;
	for i := 0; i < len(s); {
		switch s[i] {
		case '%':
			n++;
			if i+2 >= len(s) || !ishex(s[i+1]) || !ishex(s[i+2]) {
				return "", BadURL{"invalid hexadecimal escape"}
			}
			i += 3;
		case '+':
			anyPlusses = true;
			i++;
		default:
			i++
		}
	}

	if n == 0 && !anyPlusses {
		return s, nil
	}

	t := make([]byte, len(s)-2*n);
	j := 0;
	for i := 0; i < len(s); {
		switch s[i] {
		case '%':
			t[j] = unhex(s[i+1]) << 4 | unhex(s[i+2]);
			j++;
			i += 3;
		case '+':
			t[j] = ' ';
			j++;
			i++;
		default:
			t[j] = s[i];
			j++;
			i++;
		}
	}
	return string(t), nil;
}

// URLEscape converts a string into URL-encoded form.
func URLEscape(s string) string {
	spaceCount, hexCount := 0, 0;
	for i := 0; i < len(s); i++ {
		c := s[i];
		if (shouldEscape(c)) {
			if (c == ' ') {
				spaceCount++;
			} else {
				hexCount++;
			}
		}
	}

	if spaceCount == 0 && hexCount == 0 {
		return s;
	}

	t := make([]byte, len(s)+2*hexCount);
	j := 0;
	for i := 0; i < len(s); i++ {
		c := s[i];
		if !shouldEscape(c) {
			t[j] = s[i];
			j++;
		} else if (c == ' ') {
			t[j] = '+';
			j++;
		} else {
			t[j] = '%';
			t[j+1] = "0123456789abcdef"[c>>4];
			t[j+2] = "0123456789abcdef"[c&15];
			j += 3;
		}
	}
	return string(t);
}

// A URL represents a parsed URL (technically, a URI reference).
// The general form represented is:
//	scheme://[userinfo@]host/path[?query][#fragment]
// The Raw, RawPath, and RawQuery fields are in "wire format" (special
// characters must be hex-escaped if not meant to have special meaning).
// All other fields are logical values; '+' or '%' represent themselves.
//
// Note, the reason for using wire format for the query is that it needs
// to be split into key/value pairs before decoding.
type URL struct {
	Raw string;		// the original string
	Scheme string;		// scheme
	RawPath string;		// //[userinfo@]host/path[?query][#fragment]
	Authority string;	// [userinfo@]host
	Userinfo string;	// userinfo
	Host string;		// host
	Path string;		// /path
	RawQuery string;	// query
	Fragment string;	// fragment
}

// Maybe rawurl is of the form scheme:path.
// (Scheme must be [a-zA-Z][a-zA-Z0-9+-.]*)
// If so, return scheme, path; else return "", rawurl.
func getscheme(rawurl string) (scheme, path string, err os.Error) {
	for i := 0; i < len(rawurl); i++ {
		c := rawurl[i];
		switch {
		case 'a' <= c && c <= 'z' ||'A' <= c && c <= 'Z':
			// do nothing
		case '0' <= c && c <= '9' || c == '+' || c == '-' || c == '.':
			if i == 0 {
				return "", rawurl, nil
			}
		case c == ':':
			if i == 0 {
				return "", "", BadURL{"missing protocol scheme"}
			}
			return rawurl[0:i], rawurl[i+1:len(rawurl)], nil
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
				return s[0:i], s[i+1:len(s)]
			}
			return s[0:i], s[i:len(s)]
		}
	}
	return s, ""
}

// BUG(rsc): ParseURL should canonicalize the path,
// removing unnecessary . and .. elements.

// ParseURL parses rawurl into a URL structure.
// The string rawurl is assumed not to have a #fragment suffix.
// (Web browsers strip #fragment before sending the URL to a web server.)
func ParseURL(rawurl string) (url *URL, err os.Error) {
	if rawurl == "" {
		return nil, BadURL{"empty url"}
	}
	url = new(URL);
	url.Raw = rawurl;

	// split off possible leading "http:", "mailto:", etc.
	var path string;
	if url.Scheme, path, err = getscheme(rawurl); err != nil {
		return nil, err
	}
	url.RawPath = path;

	// RFC 2396: a relative URI (no scheme) has a ?query,
	// but absolute URIs only have query if path begins with /
	if url.Scheme == "" || len(path) > 0 && path[0] == '/' {
		path, url.RawQuery = split(path, '?', true);
	}

	// Maybe path is //authority/path
	if len(path) > 2 && path[0:2] == "//" {
		url.Authority, path = split(path[2:len(path)], '/', false);
	}

	// If there's no @, split's default is wrong.  Check explicitly.
	if strings.Index(url.Authority, "@") < 0 {
		url.Host = url.Authority;
	} else {
		url.Userinfo, url.Host = split(url.Authority, '@', true);
	}

	// What's left is the path.
	// TODO: Canonicalize (remove . and ..)?
	if url.Path, err = URLUnescape(path); err != nil {
		return nil, err
	}

	// Remove escapes from the Authority and Userinfo fields, and verify
	// that Scheme and Host contain no escapes (that would be illegal).
	if url.Authority, err = URLUnescape(url.Authority); err != nil {
		return nil, err
	}
	if url.Userinfo, err = URLUnescape(url.Userinfo); err != nil {
		return nil, err
	}
	if (strings.Index(url.Scheme, "%") >= 0) {
		return nil, BadURL{"hexadecimal escape in scheme"}
	}
	if (strings.Index(url.Host, "%") >= 0) {
		return nil, BadURL{"hexadecimal escape in host"}
	}

	return url, nil
}

// ParseURLReference is like ParseURL but allows a trailing #fragment.
func ParseURLReference(rawurlref string) (url *URL, err os.Error) {
	// Cut off #frag.
	rawurl, frag := split(rawurlref, '#', true);
	if url, err = ParseURL(rawurl); err != nil {
		return nil, err
	}
	if url.Fragment, err = URLUnescape(frag); err != nil {
		return nil, err
	}
	return url, nil
}

// String reassembles url into a valid URL string.
//
// There are redundant fields stored in the URL structure:
// the String method consults Scheme, Path, Host, Userinfo,
// RawQuery, and Fragment, but not Raw, RawPath or Authority.
func (url *URL) String() string {
	result := "";
	if url.Scheme != "" {
		result += url.Scheme + ":";
	}
	if url.Host != "" || url.Userinfo != "" {
		result += "//";
		if url.Userinfo != "" {
			result += URLEscape(url.Userinfo) + "@";
		}
		result += url.Host;
	}
	result += URLEscape(url.Path);
	if url.RawQuery != "" {
		result += "?" + url.RawQuery;
	}
	if url.Fragment != "" {
		result += "#" + URLEscape(url.Fragment);
	}
	return result;
}
