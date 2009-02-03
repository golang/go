// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parse URLs (actually URIs, but that seems overly pedantic).
// TODO(rsc): Add tests.

package http

import (
	"os";
	"strings"
)

var (
	BadURL = os.NewError("bad url syntax")
)

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

// Unescape %xx into hex.
func URLUnescape(s string) (string, *os.Error) {
	// Count %, check that they're well-formed.
	n := 0;
	for i := 0; i < len(s); {
		if s[i] == '%' {
			n++;
			if !ishex(s[i+1]) || !ishex(s[i+2]) {
				return "", BadURL;
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

type URL struct {
	Raw string;
	Scheme string;
	RawPath string;
	Authority string;
	Userinfo string;
	Host string;
	Path string;
	Query string;
	Fragment string;
}

// Maybe rawurl is of the form scheme:path.
// (Scheme must be [a-zA-Z][a-zA-Z0-9+-.]*)
// If so, return scheme, path; else return "", rawurl.
func getscheme(rawurl string) (scheme, path string, err *os.Error) {
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
				return "", "", BadURL
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

// Parse rawurl into a URL structure.
func ParseURL(rawurl string) (url *URL, err *os.Error) {
	if rawurl == "" {
		return nil, BadURL
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

// A URL reference is a URL with #frag potentially added.  Parse it.
func ParseURLReference(rawurlref string) (url *URL, err *os.Error) {
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

