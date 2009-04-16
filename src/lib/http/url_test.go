// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"fmt";
	"http";
	"os";
	"reflect";
	"testing";
)

// TODO(rsc):
//	test URLUnescape
// 	test URLEscape
//	test ParseURL

type URLTest struct {
	in string;
	out *URL;
}

var urltests = []URLTest {
	// no path
	URLTest{
		"http://www.google.com",
		&URL{
			"http://www.google.com",
			"http", "//www.google.com",
			"www.google.com", "", "www.google.com",
			"", "", ""
		}
	},
	// path
	URLTest{
		"http://www.google.com/",
		&URL{
			"http://www.google.com/",
			"http", "//www.google.com/",
			"www.google.com", "", "www.google.com",
			"/", "", ""
		}
	},
	// user
	URLTest{
		"ftp://webmaster@www.google.com/",
		&URL{
			"ftp://webmaster@www.google.com/",
			"ftp", "//webmaster@www.google.com/",
			"webmaster@www.google.com", "webmaster", "www.google.com",
			"/", "", ""
		}
	},
	// query
	URLTest{
		"http://www.google.com/?q=go+language",
		&URL{
			"http://www.google.com/?q=go+language",
			"http", "//www.google.com/?q=go+language",
			"www.google.com", "", "www.google.com",
			"/", "q=go+language", ""
		}
	},
	// path without /, so no query parsing
	URLTest{
		"http:www.google.com/?q=go+language",
		&URL{
			"http:www.google.com/?q=go+language",
			"http", "www.google.com/?q=go+language",
			"", "", "",
			"www.google.com/?q=go+language", "", ""
		}
	},
	// non-authority
	URLTest{
		"mailto:/webmaster@golang.org",
		&URL{
			"mailto:/webmaster@golang.org",
			"mailto", "/webmaster@golang.org",
			"", "", "",
			"/webmaster@golang.org", "", ""
		}
	},
	// non-authority
	URLTest{
		"mailto:webmaster@golang.org",
		&URL{
			"mailto:webmaster@golang.org",
			"mailto", "webmaster@golang.org",
			"", "", "",
			"webmaster@golang.org", "", ""
		}
	},
}

var urlnofragtests = []URLTest {
	URLTest{
		"http://www.google.com/?q=go+language#foo",
		&URL{
			"http://www.google.com/?q=go+language#foo",
			"http", "//www.google.com/?q=go+language#foo",
			"www.google.com", "", "www.google.com",
			"/", "q=go+language#foo", ""
		}
	},
}

var urlfragtests = []URLTest {
	URLTest{
		"http://www.google.com/?q=go+language#foo",
		&URL{
			"http://www.google.com/?q=go+language",
			"http", "//www.google.com/?q=go+language",
			"www.google.com", "", "www.google.com",
			"/", "q=go+language", "foo"
		}
	},
}

// more useful string for debugging than fmt's struct printer
func ufmt(u *URL) string {
	return fmt.Sprintf("%q, %q, %q, %q, %q, %q, %q, %q, %q",
		u.Raw, u.Scheme, u.RawPath, u.Authority, u.Userinfo,
		u.Host, u.Path, u.Query, u.Fragment);
}

func DoTest(t *testing.T, parse func(string) (*URL, *os.Error), name string, tests []URLTest) {
	for i, tt := range tests {
		u, err := parse(tt.in);
		if err != nil {
			t.Errorf("%s(%q) returned error %s", name, tt.in, err);
			continue;
		}
		if !reflect.DeepEqual(u, tt.out) {
			t.Errorf("%s(%q):\n\thave %v\n\twant %v\n",
				name, tt.in, ufmt(u), ufmt(tt.out));
		}
	}
}

func TestParseURL(t *testing.T) {
	DoTest(t, ParseURL, "ParseURL", urltests);
	DoTest(t, ParseURL, "ParseURL", urlnofragtests);
}

func TestParseURLReference(t *testing.T) {
	DoTest(t, ParseURLReference, "ParseURLReference", urltests);
	DoTest(t, ParseURLReference, "ParseURLReference", urlfragtests);
}

func DoTestString(t *testing.T, parse func(string) (*URL, *os.Error), name string, tests []URLTest) {
	for i, tt := range tests {
		u, err := parse(tt.in);
		if err != nil {
			t.Errorf("%s(%q) returned error %s", name, tt.in, err);
			continue;
		}
		s := u.String();
		if s != tt.in {
			t.Errorf("%s(%q).String() == %q", tt.in, s);
		}
	}
}

func TestURLString(t *testing.T) {
	DoTestString(t, ParseURL, "ParseURL", urltests);
	DoTestString(t, ParseURL, "ParseURL", urlfragtests);
	DoTestString(t, ParseURL, "ParseURL", urlnofragtests);
	DoTestString(t, ParseURLReference, "ParseURLReference", urltests);
	DoTestString(t, ParseURLReference, "ParseURLReference", urlfragtests);
	DoTestString(t, ParseURLReference, "ParseURLReference", urlnofragtests);
}
