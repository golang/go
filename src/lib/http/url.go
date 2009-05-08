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

// URLUnescape unescapes a URL-encoded string,
// converting %AB into the byte 0xAB.
// It returns a BadURL error if any % is not followed
// by two hexadecimal digits.
func URLUnescape(s string) (string, os.Error) {
	// Count %, check that they're well-formed.
	n := 0;
	for i := 0; i < len(s); {
		if s[i] == '%' {
			n++;
			if !ishex(s[i+1]) || !ishex(s[i+2]) {
				return "", BadURL{"invalid hexadecimal escape"}
			}
			i += 3
		} else {
			i++
		}
	}

	if n == 0 {
		return s, nil
	}

	t := make([]byte, len(s)-2*n);
	j := 0;
	for i := 0; i < len(s); {
		if s[i] == '%' {
			t[j] = unhex(s[i+1]) << 4 | unhex(s[i+2]);
			j++;
			i += 3;
		} else {
			t[j] = s[i];
			j++;
			i++;
		}
	}
	return string(t), nil;
}

// A URL represents a parsed URL (technically, a URI reference).
// The general form represented is:
//	scheme://[userinfo@]host/path[?query][#fragment]
type URL struct {
	Raw string;		// the original string
	Scheme string;		// scheme
	RawPath string;		// //[userinfo@]host/path[?query][#fragment]
	Authority string;	// [userinfo@]host
	Userinfo string;	// userinfo
	Host string;		// host
	Path string;		// /path
	Query string;		// query
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
		path, url.Query = split(path, '?', true);
		if url.Query, err = URLUnescape(url.Query); err != nil {
			return nil, err
		}
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
// Query, and Fragment, but not RawPath or Authority.
func (url *URL) String() string {
	result := "";
	if url.Scheme != "" {
		result += url.Scheme + ":";
	}
	if url.Host != "" || url.Userinfo != "" {
		result += "//";
		if url.Userinfo != "" {
			result += url.Userinfo + "@";
		}
		result += url.Host;
	}
	result += url.Path;
	if url.Query != "" {
		result += "?" + url.Query;
	}
	if url.Fragment != "" {
		result += "#" + url.Fragment;
	}
	return result;
}
