// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package url

import (
	"bytes"
	encodingPkg "encoding"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"reflect"
	"strings"
	"testing"
)

type URLTest struct {
	in        string
	out       *URL   // expected parse
	roundtrip string // expected result of reserializing the URL; empty means same as "in".
}

var urltests = []URLTest{
	// no path
	{
		"http://www.google.com",
		&URL{
			Scheme: "http",
			Host:   "www.google.com",
		},
		"",
	},
	// path
	{
		"http://www.google.com/",
		&URL{
			Scheme: "http",
			Host:   "www.google.com",
			Path:   "/",
		},
		"",
	},
	// path with hex escaping
	{
		"http://www.google.com/file%20one%26two",
		&URL{
			Scheme:  "http",
			Host:    "www.google.com",
			Path:    "/file one&two",
			RawPath: "/file%20one%26two",
		},
		"",
	},
	// fragment with hex escaping
	{
		"http://www.google.com/#file%20one%26two",
		&URL{
			Scheme:      "http",
			Host:        "www.google.com",
			Path:        "/",
			Fragment:    "file one&two",
			RawFragment: "file%20one%26two",
		},
		"",
	},
	// user
	{
		"ftp://webmaster@www.google.com/",
		&URL{
			Scheme: "ftp",
			User:   User("webmaster"),
			Host:   "www.google.com",
			Path:   "/",
		},
		"",
	},
	// escape sequence in username
	{
		"ftp://john%20doe@www.google.com/",
		&URL{
			Scheme: "ftp",
			User:   User("john doe"),
			Host:   "www.google.com",
			Path:   "/",
		},
		"ftp://john%20doe@www.google.com/",
	},
	// empty query
	{
		"http://www.google.com/?",
		&URL{
			Scheme:     "http",
			Host:       "www.google.com",
			Path:       "/",
			ForceQuery: true,
		},
		"",
	},
	// query ending in question mark (Issue 14573)
	{
		"http://www.google.com/?foo=bar?",
		&URL{
			Scheme:   "http",
			Host:     "www.google.com",
			Path:     "/",
			RawQuery: "foo=bar?",
		},
		"",
	},
	// query
	{
		"http://www.google.com/?q=go+language",
		&URL{
			Scheme:   "http",
			Host:     "www.google.com",
			Path:     "/",
			RawQuery: "q=go+language",
		},
		"",
	},
	// query with hex escaping: NOT parsed
	{
		"http://www.google.com/?q=go%20language",
		&URL{
			Scheme:   "http",
			Host:     "www.google.com",
			Path:     "/",
			RawQuery: "q=go%20language",
		},
		"",
	},
	// %20 outside query
	{
		"http://www.google.com/a%20b?q=c+d",
		&URL{
			Scheme:   "http",
			Host:     "www.google.com",
			Path:     "/a b",
			RawQuery: "q=c+d",
		},
		"",
	},
	// path without leading /, so no parsing
	{
		"http:www.google.com/?q=go+language",
		&URL{
			Scheme:   "http",
			Opaque:   "www.google.com/",
			RawQuery: "q=go+language",
		},
		"http:www.google.com/?q=go+language",
	},
	// path without leading /, so no parsing
	{
		"http:%2f%2fwww.google.com/?q=go+language",
		&URL{
			Scheme:   "http",
			Opaque:   "%2f%2fwww.google.com/",
			RawQuery: "q=go+language",
		},
		"http:%2f%2fwww.google.com/?q=go+language",
	},
	// non-authority with path
	{
		"mailto:/webmaster@golang.org",
		&URL{
			Scheme: "mailto",
			Path:   "/webmaster@golang.org",
		},
		"mailto:///webmaster@golang.org", // unfortunate compromise
	},
	// non-authority
	{
		"mailto:webmaster@golang.org",
		&URL{
			Scheme: "mailto",
			Opaque: "webmaster@golang.org",
		},
		"",
	},
	// unescaped :// in query should not create a scheme
	{
		"/foo?query=http://bad",
		&URL{
			Path:     "/foo",
			RawQuery: "query=http://bad",
		},
		"",
	},
	// leading // without scheme should create an authority
	{
		"//foo",
		&URL{
			Host: "foo",
		},
		"",
	},
	// leading // without scheme, with userinfo, path, and query
	{
		"//user@foo/path?a=b",
		&URL{
			User:     User("user"),
			Host:     "foo",
			Path:     "/path",
			RawQuery: "a=b",
		},
		"",
	},
	// Three leading slashes isn't an authority, but doesn't return an error.
	// (We can't return an error, as this code is also used via
	// ServeHTTP -> ReadRequest -> Parse, which is arguably a
	// different URL parsing context, but currently shares the
	// same codepath)
	{
		"///threeslashes",
		&URL{
			Path: "///threeslashes",
		},
		"",
	},
	{
		"http://user:password@google.com",
		&URL{
			Scheme: "http",
			User:   UserPassword("user", "password"),
			Host:   "google.com",
		},
		"http://user:password@google.com",
	},
	// unescaped @ in username should not confuse host
	{
		"http://j@ne:password@google.com",
		&URL{
			Scheme: "http",
			User:   UserPassword("j@ne", "password"),
			Host:   "google.com",
		},
		"http://j%40ne:password@google.com",
	},
	// unescaped @ in password should not confuse host
	{
		"http://jane:p@ssword@google.com",
		&URL{
			Scheme: "http",
			User:   UserPassword("jane", "p@ssword"),
			Host:   "google.com",
		},
		"http://jane:p%40ssword@google.com",
	},
	{
		"http://j@ne:password@google.com/p@th?q=@go",
		&URL{
			Scheme:   "http",
			User:     UserPassword("j@ne", "password"),
			Host:     "google.com",
			Path:     "/p@th",
			RawQuery: "q=@go",
		},
		"http://j%40ne:password@google.com/p@th?q=@go",
	},
	{
		"http://www.google.com/?q=go+language#foo",
		&URL{
			Scheme:   "http",
			Host:     "www.google.com",
			Path:     "/",
			RawQuery: "q=go+language",
			Fragment: "foo",
		},
		"",
	},
	{
		"http://www.google.com/?q=go+language#foo&bar",
		&URL{
			Scheme:   "http",
			Host:     "www.google.com",
			Path:     "/",
			RawQuery: "q=go+language",
			Fragment: "foo&bar",
		},
		"http://www.google.com/?q=go+language#foo&bar",
	},
	{
		"http://www.google.com/?q=go+language#foo%26bar",
		&URL{
			Scheme:      "http",
			Host:        "www.google.com",
			Path:        "/",
			RawQuery:    "q=go+language",
			Fragment:    "foo&bar",
			RawFragment: "foo%26bar",
		},
		"http://www.google.com/?q=go+language#foo%26bar",
	},
	{
		"file:///home/adg/rabbits",
		&URL{
			Scheme: "file",
			Host:   "",
			Path:   "/home/adg/rabbits",
		},
		"file:///home/adg/rabbits",
	},
	// "Windows" paths are no exception to the rule.
	// See golang.org/issue/6027, especially comment #9.
	{
		"file:///C:/FooBar/Baz.txt",
		&URL{
			Scheme: "file",
			Host:   "",
			Path:   "/C:/FooBar/Baz.txt",
		},
		"file:///C:/FooBar/Baz.txt",
	},
	// case-insensitive scheme
	{
		"MaIlTo:webmaster@golang.org",
		&URL{
			Scheme: "mailto",
			Opaque: "webmaster@golang.org",
		},
		"mailto:webmaster@golang.org",
	},
	// Relative path
	{
		"a/b/c",
		&URL{
			Path: "a/b/c",
		},
		"a/b/c",
	},
	// escaped '?' in username and password
	{
		"http://%3Fam:pa%3Fsword@google.com",
		&URL{
			Scheme: "http",
			User:   UserPassword("?am", "pa?sword"),
			Host:   "google.com",
		},
		"",
	},
	// host subcomponent; IPv4 address in RFC 3986
	{
		"http://192.168.0.1/",
		&URL{
			Scheme: "http",
			Host:   "192.168.0.1",
			Path:   "/",
		},
		"",
	},
	// host and port subcomponents; IPv4 address in RFC 3986
	{
		"http://192.168.0.1:8080/",
		&URL{
			Scheme: "http",
			Host:   "192.168.0.1:8080",
			Path:   "/",
		},
		"",
	},
	// host subcomponent; IPv6 address in RFC 3986
	{
		"http://[fe80::1]/",
		&URL{
			Scheme: "http",
			Host:   "[fe80::1]",
			Path:   "/",
		},
		"",
	},
	// host and port subcomponents; IPv6 address in RFC 3986
	{
		"http://[fe80::1]:8080/",
		&URL{
			Scheme: "http",
			Host:   "[fe80::1]:8080",
			Path:   "/",
		},
		"",
	},
	// host subcomponent; IPv6 address with zone identifier in RFC 6874
	{
		"http://[fe80::1%25en0]/", // alphanum zone identifier
		&URL{
			Scheme: "http",
			Host:   "[fe80::1%en0]",
			Path:   "/",
		},
		"",
	},
	// host and port subcomponents; IPv6 address with zone identifier in RFC 6874
	{
		"http://[fe80::1%25en0]:8080/", // alphanum zone identifier
		&URL{
			Scheme: "http",
			Host:   "[fe80::1%en0]:8080",
			Path:   "/",
		},
		"",
	},
	// host subcomponent; IPv6 address with zone identifier in RFC 6874
	{
		"http://[fe80::1%25%65%6e%301-._~]/", // percent-encoded+unreserved zone identifier
		&URL{
			Scheme: "http",
			Host:   "[fe80::1%en01-._~]",
			Path:   "/",
		},
		"http://[fe80::1%25en01-._~]/",
	},
	// host and port subcomponents; IPv6 address with zone identifier in RFC 6874
	{
		"http://[fe80::1%25%65%6e%301-._~]:8080/", // percent-encoded+unreserved zone identifier
		&URL{
			Scheme: "http",
			Host:   "[fe80::1%en01-._~]:8080",
			Path:   "/",
		},
		"http://[fe80::1%25en01-._~]:8080/",
	},
	// alternate escapings of path survive round trip
	{
		"http://rest.rsc.io/foo%2fbar/baz%2Fquux?alt=media",
		&URL{
			Scheme:   "http",
			Host:     "rest.rsc.io",
			Path:     "/foo/bar/baz/quux",
			RawPath:  "/foo%2fbar/baz%2Fquux",
			RawQuery: "alt=media",
		},
		"",
	},
	// issue 12036
	{
		"mysql://a,b,c/bar",
		&URL{
			Scheme: "mysql",
			Host:   "a,b,c",
			Path:   "/bar",
		},
		"",
	},
	// worst case host, still round trips
	{
		"scheme://!$&'()*+,;=hello!:1/path",
		&URL{
			Scheme: "scheme",
			Host:   "!$&'()*+,;=hello!:1",
			Path:   "/path",
		},
		"",
	},
	// worst case path, still round trips
	{
		"http://host/!$&'()*+,;=:@[hello]",
		&URL{
			Scheme:  "http",
			Host:    "host",
			Path:    "/!$&'()*+,;=:@[hello]",
			RawPath: "/!$&'()*+,;=:@[hello]",
		},
		"",
	},
	// golang.org/issue/5684
	{
		"http://example.com/oid/[order_id]",
		&URL{
			Scheme:  "http",
			Host:    "example.com",
			Path:    "/oid/[order_id]",
			RawPath: "/oid/[order_id]",
		},
		"",
	},
	// golang.org/issue/12200 (colon with empty port)
	{
		"http://192.168.0.2:8080/foo",
		&URL{
			Scheme: "http",
			Host:   "192.168.0.2:8080",
			Path:   "/foo",
		},
		"",
	},
	{
		"http://192.168.0.2:/foo",
		&URL{
			Scheme: "http",
			Host:   "192.168.0.2:",
			Path:   "/foo",
		},
		"",
	},
	{
		// Malformed IPv6 but still accepted.
		"http://2b01:e34:ef40:7730:8e70:5aff:fefe:edac:8080/foo",
		&URL{
			Scheme: "http",
			Host:   "2b01:e34:ef40:7730:8e70:5aff:fefe:edac:8080",
			Path:   "/foo",
		},
		"",
	},
	{
		// Malformed IPv6 but still accepted.
		"http://2b01:e34:ef40:7730:8e70:5aff:fefe:edac:/foo",
		&URL{
			Scheme: "http",
			Host:   "2b01:e34:ef40:7730:8e70:5aff:fefe:edac:",
			Path:   "/foo",
		},
		"",
	},
	{
		"http://[2b01:e34:ef40:7730:8e70:5aff:fefe:edac]:8080/foo",
		&URL{
			Scheme: "http",
			Host:   "[2b01:e34:ef40:7730:8e70:5aff:fefe:edac]:8080",
			Path:   "/foo",
		},
		"",
	},
	{
		"http://[2b01:e34:ef40:7730:8e70:5aff:fefe:edac]:/foo",
		&URL{
			Scheme: "http",
			Host:   "[2b01:e34:ef40:7730:8e70:5aff:fefe:edac]:",
			Path:   "/foo",
		},
		"",
	},
	// golang.org/issue/7991 and golang.org/issue/12719 (non-ascii %-encoded in host)
	{
		"http://hello.世界.com/foo",
		&URL{
			Scheme: "http",
			Host:   "hello.世界.com",
			Path:   "/foo",
		},
		"http://hello.%E4%B8%96%E7%95%8C.com/foo",
	},
	{
		"http://hello.%e4%b8%96%e7%95%8c.com/foo",
		&URL{
			Scheme: "http",
			Host:   "hello.世界.com",
			Path:   "/foo",
		},
		"http://hello.%E4%B8%96%E7%95%8C.com/foo",
	},
	{
		"http://hello.%E4%B8%96%E7%95%8C.com/foo",
		&URL{
			Scheme: "http",
			Host:   "hello.世界.com",
			Path:   "/foo",
		},
		"",
	},
	// golang.org/issue/10433 (path beginning with //)
	{
		"http://example.com//foo",
		&URL{
			Scheme: "http",
			Host:   "example.com",
			Path:   "//foo",
		},
		"",
	},
	// test that we can reparse the host names we accept.
	{
		"myscheme://authority<\"hi\">/foo",
		&URL{
			Scheme: "myscheme",
			Host:   "authority<\"hi\">",
			Path:   "/foo",
		},
		"",
	},
	// spaces in hosts are disallowed but escaped spaces in IPv6 scope IDs are grudgingly OK.
	// This happens on Windows.
	// golang.org/issue/14002
	{
		"tcp://[2020::2020:20:2020:2020%25Windows%20Loves%20Spaces]:2020",
		&URL{
			Scheme: "tcp",
			Host:   "[2020::2020:20:2020:2020%Windows Loves Spaces]:2020",
		},
		"",
	},
	// test we can roundtrip magnet url
	// fix issue https://golang.org/issue/20054
	{
		"magnet:?xt=urn:btih:c12fe1c06bba254a9dc9f519b335aa7c1367a88a&dn",
		&URL{
			Scheme:   "magnet",
			Host:     "",
			Path:     "",
			RawQuery: "xt=urn:btih:c12fe1c06bba254a9dc9f519b335aa7c1367a88a&dn",
		},
		"magnet:?xt=urn:btih:c12fe1c06bba254a9dc9f519b335aa7c1367a88a&dn",
	},
	{
		"mailto:?subject=hi",
		&URL{
			Scheme:   "mailto",
			Host:     "",
			Path:     "",
			RawQuery: "subject=hi",
		},
		"mailto:?subject=hi",
	},
}

// more useful string for debugging than fmt's struct printer
func ufmt(u *URL) string {
	var user, pass interface{}
	if u.User != nil {
		user = u.User.Username()
		if p, ok := u.User.Password(); ok {
			pass = p
		}
	}
	return fmt.Sprintf("opaque=%q, scheme=%q, user=%#v, pass=%#v, host=%q, path=%q, rawpath=%q, rawq=%q, frag=%q, rawfrag=%q, forcequery=%v",
		u.Opaque, u.Scheme, user, pass, u.Host, u.Path, u.RawPath, u.RawQuery, u.Fragment, u.RawFragment, u.ForceQuery)
}

func BenchmarkString(b *testing.B) {
	b.StopTimer()
	b.ReportAllocs()
	for _, tt := range urltests {
		u, err := Parse(tt.in)
		if err != nil {
			b.Errorf("Parse(%q) returned error %s", tt.in, err)
			continue
		}
		if tt.roundtrip == "" {
			continue
		}
		b.StartTimer()
		var g string
		for i := 0; i < b.N; i++ {
			g = u.String()
		}
		b.StopTimer()
		if w := tt.roundtrip; b.N > 0 && g != w {
			b.Errorf("Parse(%q).String() == %q, want %q", tt.in, g, w)
		}
	}
}

func TestParse(t *testing.T) {
	for _, tt := range urltests {
		u, err := Parse(tt.in)
		if err != nil {
			t.Errorf("Parse(%q) returned error %v", tt.in, err)
			continue
		}
		if !reflect.DeepEqual(u, tt.out) {
			t.Errorf("Parse(%q):\n\tgot  %v\n\twant %v\n", tt.in, ufmt(u), ufmt(tt.out))
		}
	}
}

const pathThatLooksSchemeRelative = "//not.a.user@not.a.host/just/a/path"

var parseRequestURLTests = []struct {
	url           string
	expectedValid bool
}{
	{"http://foo.com", true},
	{"http://foo.com/", true},
	{"http://foo.com/path", true},
	{"/", true},
	{pathThatLooksSchemeRelative, true},
	{"//not.a.user@%66%6f%6f.com/just/a/path/also", true},
	{"*", true},
	{"http://192.168.0.1/", true},
	{"http://192.168.0.1:8080/", true},
	{"http://[fe80::1]/", true},
	{"http://[fe80::1]:8080/", true},

	// Tests exercising RFC 6874 compliance:
	{"http://[fe80::1%25en0]/", true},                 // with alphanum zone identifier
	{"http://[fe80::1%25en0]:8080/", true},            // with alphanum zone identifier
	{"http://[fe80::1%25%65%6e%301-._~]/", true},      // with percent-encoded+unreserved zone identifier
	{"http://[fe80::1%25%65%6e%301-._~]:8080/", true}, // with percent-encoded+unreserved zone identifier

	{"foo.html", false},
	{"../dir/", false},
	{" http://foo.com", false},
	{"http://192.168.0.%31/", false},
	{"http://192.168.0.%31:8080/", false},
	{"http://[fe80::%31]/", false},
	{"http://[fe80::%31]:8080/", false},
	{"http://[fe80::%31%25en0]/", false},
	{"http://[fe80::%31%25en0]:8080/", false},

	// These two cases are valid as textual representations as
	// described in RFC 4007, but are not valid as address
	// literals with IPv6 zone identifiers in URIs as described in
	// RFC 6874.
	{"http://[fe80::1%en0]/", false},
	{"http://[fe80::1%en0]:8080/", false},
}

func TestParseRequestURI(t *testing.T) {
	for _, test := range parseRequestURLTests {
		_, err := ParseRequestURI(test.url)
		if test.expectedValid && err != nil {
			t.Errorf("ParseRequestURI(%q) gave err %v; want no error", test.url, err)
		} else if !test.expectedValid && err == nil {
			t.Errorf("ParseRequestURI(%q) gave nil error; want some error", test.url)
		}
	}

	url, err := ParseRequestURI(pathThatLooksSchemeRelative)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if url.Path != pathThatLooksSchemeRelative {
		t.Errorf("ParseRequestURI path:\ngot  %q\nwant %q", url.Path, pathThatLooksSchemeRelative)
	}
}

var stringURLTests = []struct {
	url  URL
	want string
}{
	// No leading slash on path should prepend slash on String() call
	{
		url: URL{
			Scheme: "http",
			Host:   "www.google.com",
			Path:   "search",
		},
		want: "http://www.google.com/search",
	},
	// Relative path with first element containing ":" should be prepended with "./", golang.org/issue/17184
	{
		url: URL{
			Path: "this:that",
		},
		want: "./this:that",
	},
	// Relative path with second element containing ":" should not be prepended with "./"
	{
		url: URL{
			Path: "here/this:that",
		},
		want: "here/this:that",
	},
	// Non-relative path with first element containing ":" should not be prepended with "./"
	{
		url: URL{
			Scheme: "http",
			Host:   "www.google.com",
			Path:   "this:that",
		},
		want: "http://www.google.com/this:that",
	},
}

func TestURLString(t *testing.T) {
	for _, tt := range urltests {
		u, err := Parse(tt.in)
		if err != nil {
			t.Errorf("Parse(%q) returned error %s", tt.in, err)
			continue
		}
		expected := tt.in
		if tt.roundtrip != "" {
			expected = tt.roundtrip
		}
		s := u.String()
		if s != expected {
			t.Errorf("Parse(%q).String() == %q (expected %q)", tt.in, s, expected)
		}
	}

	for _, tt := range stringURLTests {
		if got := tt.url.String(); got != tt.want {
			t.Errorf("%+v.String() = %q; want %q", tt.url, got, tt.want)
		}
	}
}

func TestURLRedacted(t *testing.T) {
	cases := []struct {
		name string
		url  *URL
		want string
	}{
		{
			name: "non-blank Password",
			url: &URL{
				Scheme: "http",
				Host:   "host.tld",
				Path:   "this:that",
				User:   UserPassword("user", "password"),
			},
			want: "http://user:xxxxx@host.tld/this:that",
		},
		{
			name: "blank Password",
			url: &URL{
				Scheme: "http",
				Host:   "host.tld",
				Path:   "this:that",
				User:   User("user"),
			},
			want: "http://user@host.tld/this:that",
		},
		{
			name: "nil User",
			url: &URL{
				Scheme: "http",
				Host:   "host.tld",
				Path:   "this:that",
				User:   UserPassword("", "password"),
			},
			want: "http://:xxxxx@host.tld/this:that",
		},
		{
			name: "blank Username, blank Password",
			url: &URL{
				Scheme: "http",
				Host:   "host.tld",
				Path:   "this:that",
			},
			want: "http://host.tld/this:that",
		},
		{
			name: "empty URL",
			url:  &URL{},
			want: "",
		},
		{
			name: "nil URL",
			url:  nil,
			want: "",
		},
	}

	for _, tt := range cases {
		t := t
		t.Run(tt.name, func(t *testing.T) {
			if g, w := tt.url.Redacted(), tt.want; g != w {
				t.Fatalf("got: %q\nwant: %q", g, w)
			}
		})
	}
}

type EscapeTest struct {
	in  string
	out string
	err error
}

var unescapeTests = []EscapeTest{
	{
		"",
		"",
		nil,
	},
	{
		"abc",
		"abc",
		nil,
	},
	{
		"1%41",
		"1A",
		nil,
	},
	{
		"1%41%42%43",
		"1ABC",
		nil,
	},
	{
		"%4a",
		"J",
		nil,
	},
	{
		"%6F",
		"o",
		nil,
	},
	{
		"%", // not enough characters after %
		"",
		EscapeError("%"),
	},
	{
		"%a", // not enough characters after %
		"",
		EscapeError("%a"),
	},
	{
		"%1", // not enough characters after %
		"",
		EscapeError("%1"),
	},
	{
		"123%45%6", // not enough characters after %
		"",
		EscapeError("%6"),
	},
	{
		"%zzzzz", // invalid hex digits
		"",
		EscapeError("%zz"),
	},
	{
		"a+b",
		"a b",
		nil,
	},
	{
		"a%20b",
		"a b",
		nil,
	},
}

func TestUnescape(t *testing.T) {
	for _, tt := range unescapeTests {
		actual, err := QueryUnescape(tt.in)
		if actual != tt.out || (err != nil) != (tt.err != nil) {
			t.Errorf("QueryUnescape(%q) = %q, %s; want %q, %s", tt.in, actual, err, tt.out, tt.err)
		}

		in := tt.in
		out := tt.out
		if strings.Contains(tt.in, "+") {
			in = strings.ReplaceAll(tt.in, "+", "%20")
			actual, err := PathUnescape(in)
			if actual != tt.out || (err != nil) != (tt.err != nil) {
				t.Errorf("PathUnescape(%q) = %q, %s; want %q, %s", in, actual, err, tt.out, tt.err)
			}
			if tt.err == nil {
				s, err := QueryUnescape(strings.ReplaceAll(tt.in, "+", "XXX"))
				if err != nil {
					continue
				}
				in = tt.in
				out = strings.ReplaceAll(s, "XXX", "+")
			}
		}

		actual, err = PathUnescape(in)
		if actual != out || (err != nil) != (tt.err != nil) {
			t.Errorf("PathUnescape(%q) = %q, %s; want %q, %s", in, actual, err, out, tt.err)
		}
	}
}

var queryEscapeTests = []EscapeTest{
	{
		"",
		"",
		nil,
	},
	{
		"abc",
		"abc",
		nil,
	},
	{
		"one two",
		"one+two",
		nil,
	},
	{
		"10%",
		"10%25",
		nil,
	},
	{
		" ?&=#+%!<>#\"{}|\\^[]`☺\t:/@$'()*,;",
		"+%3F%26%3D%23%2B%25%21%3C%3E%23%22%7B%7D%7C%5C%5E%5B%5D%60%E2%98%BA%09%3A%2F%40%24%27%28%29%2A%2C%3B",
		nil,
	},
}

func TestQueryEscape(t *testing.T) {
	for _, tt := range queryEscapeTests {
		actual := QueryEscape(tt.in)
		if tt.out != actual {
			t.Errorf("QueryEscape(%q) = %q, want %q", tt.in, actual, tt.out)
		}

		// for bonus points, verify that escape:unescape is an identity.
		roundtrip, err := QueryUnescape(actual)
		if roundtrip != tt.in || err != nil {
			t.Errorf("QueryUnescape(%q) = %q, %s; want %q, %s", actual, roundtrip, err, tt.in, "[no error]")
		}
	}
}

var pathEscapeTests = []EscapeTest{
	{
		"",
		"",
		nil,
	},
	{
		"abc",
		"abc",
		nil,
	},
	{
		"abc+def",
		"abc+def",
		nil,
	},
	{
		"a/b",
		"a%2Fb",
		nil,
	},
	{
		"one two",
		"one%20two",
		nil,
	},
	{
		"10%",
		"10%25",
		nil,
	},
	{
		" ?&=#+%!<>#\"{}|\\^[]`☺\t:/@$'()*,;",
		"%20%3F&=%23+%25%21%3C%3E%23%22%7B%7D%7C%5C%5E%5B%5D%60%E2%98%BA%09:%2F@$%27%28%29%2A%2C%3B",
		nil,
	},
}

func TestPathEscape(t *testing.T) {
	for _, tt := range pathEscapeTests {
		actual := PathEscape(tt.in)
		if tt.out != actual {
			t.Errorf("PathEscape(%q) = %q, want %q", tt.in, actual, tt.out)
		}

		// for bonus points, verify that escape:unescape is an identity.
		roundtrip, err := PathUnescape(actual)
		if roundtrip != tt.in || err != nil {
			t.Errorf("PathUnescape(%q) = %q, %s; want %q, %s", actual, roundtrip, err, tt.in, "[no error]")
		}
	}
}

//var userinfoTests = []UserinfoTest{
//	{"user", "password", "user:password"},
//	{"foo:bar", "~!@#$%^&*()_+{}|[]\\-=`:;'\"<>?,./",
//		"foo%3Abar:~!%40%23$%25%5E&*()_+%7B%7D%7C%5B%5D%5C-=%60%3A;'%22%3C%3E?,.%2F"},
//}

type EncodeQueryTest struct {
	m        Values
	expected string
}

var encodeQueryTests = []EncodeQueryTest{
	{nil, ""},
	{Values{"q": {"puppies"}, "oe": {"utf8"}}, "oe=utf8&q=puppies"},
	{Values{"q": {"dogs", "&", "7"}}, "q=dogs&q=%26&q=7"},
	{Values{
		"a": {"a1", "a2", "a3"},
		"b": {"b1", "b2", "b3"},
		"c": {"c1", "c2", "c3"},
	}, "a=a1&a=a2&a=a3&b=b1&b=b2&b=b3&c=c1&c=c2&c=c3"},
}

func TestEncodeQuery(t *testing.T) {
	for _, tt := range encodeQueryTests {
		if q := tt.m.Encode(); q != tt.expected {
			t.Errorf(`EncodeQuery(%+v) = %q, want %q`, tt.m, q, tt.expected)
		}
	}
}

var resolvePathTests = []struct {
	base, ref, expected string
}{
	{"a/b", ".", "/a/"},
	{"a/b", "c", "/a/c"},
	{"a/b", "..", "/"},
	{"a/", "..", "/"},
	{"a/", "../..", "/"},
	{"a/b/c", "..", "/a/"},
	{"a/b/c", "../d", "/a/d"},
	{"a/b/c", ".././d", "/a/d"},
	{"a/b", "./..", "/"},
	{"a/./b", ".", "/a/"},
	{"a/../", ".", "/"},
	{"a/.././b", "c", "/c"},
}

func TestResolvePath(t *testing.T) {
	for _, test := range resolvePathTests {
		got := resolvePath(test.base, test.ref)
		if got != test.expected {
			t.Errorf("For %q + %q got %q; expected %q", test.base, test.ref, got, test.expected)
		}
	}
}

var resolveReferenceTests = []struct {
	base, rel, expected string
}{
	// Absolute URL references
	{"http://foo.com?a=b", "https://bar.com/", "https://bar.com/"},
	{"http://foo.com/", "https://bar.com/?a=b", "https://bar.com/?a=b"},
	{"http://foo.com/", "https://bar.com/?", "https://bar.com/?"},
	{"http://foo.com/bar", "mailto:foo@example.com", "mailto:foo@example.com"},

	// Path-absolute references
	{"http://foo.com/bar", "/baz", "http://foo.com/baz"},
	{"http://foo.com/bar?a=b#f", "/baz", "http://foo.com/baz"},
	{"http://foo.com/bar?a=b", "/baz?", "http://foo.com/baz?"},
	{"http://foo.com/bar?a=b", "/baz?c=d", "http://foo.com/baz?c=d"},

	// Multiple slashes
	{"http://foo.com/bar", "http://foo.com//baz", "http://foo.com//baz"},
	{"http://foo.com/bar", "http://foo.com///baz/quux", "http://foo.com///baz/quux"},

	// Scheme-relative
	{"https://foo.com/bar?a=b", "//bar.com/quux", "https://bar.com/quux"},

	// Path-relative references:

	// ... current directory
	{"http://foo.com", ".", "http://foo.com/"},
	{"http://foo.com/bar", ".", "http://foo.com/"},
	{"http://foo.com/bar/", ".", "http://foo.com/bar/"},

	// ... going down
	{"http://foo.com", "bar", "http://foo.com/bar"},
	{"http://foo.com/", "bar", "http://foo.com/bar"},
	{"http://foo.com/bar/baz", "quux", "http://foo.com/bar/quux"},

	// ... going up
	{"http://foo.com/bar/baz", "../quux", "http://foo.com/quux"},
	{"http://foo.com/bar/baz", "../../../../../quux", "http://foo.com/quux"},
	{"http://foo.com/bar", "..", "http://foo.com/"},
	{"http://foo.com/bar/baz", "./..", "http://foo.com/"},
	// ".." in the middle (issue 3560)
	{"http://foo.com/bar/baz", "quux/dotdot/../tail", "http://foo.com/bar/quux/tail"},
	{"http://foo.com/bar/baz", "quux/./dotdot/../tail", "http://foo.com/bar/quux/tail"},
	{"http://foo.com/bar/baz", "quux/./dotdot/.././tail", "http://foo.com/bar/quux/tail"},
	{"http://foo.com/bar/baz", "quux/./dotdot/./../tail", "http://foo.com/bar/quux/tail"},
	{"http://foo.com/bar/baz", "quux/./dotdot/dotdot/././../../tail", "http://foo.com/bar/quux/tail"},
	{"http://foo.com/bar/baz", "quux/./dotdot/dotdot/./.././../tail", "http://foo.com/bar/quux/tail"},
	{"http://foo.com/bar/baz", "quux/./dotdot/dotdot/dotdot/./../../.././././tail", "http://foo.com/bar/quux/tail"},
	{"http://foo.com/bar/baz", "quux/./dotdot/../dotdot/../dot/./tail/..", "http://foo.com/bar/quux/dot/"},

	// Remove any dot-segments prior to forming the target URI.
	// http://tools.ietf.org/html/rfc3986#section-5.2.4
	{"http://foo.com/dot/./dotdot/../foo/bar", "../baz", "http://foo.com/dot/baz"},

	// Triple dot isn't special
	{"http://foo.com/bar", "...", "http://foo.com/..."},

	// Fragment
	{"http://foo.com/bar", ".#frag", "http://foo.com/#frag"},
	{"http://example.org/", "#!$&%27()*+,;=", "http://example.org/#!$&%27()*+,;="},

	// Paths with escaping (issue 16947).
	{"http://foo.com/foo%2fbar/", "../baz", "http://foo.com/baz"},
	{"http://foo.com/1/2%2f/3%2f4/5", "../../a/b/c", "http://foo.com/1/a/b/c"},
	{"http://foo.com/1/2/3", "./a%2f../../b/..%2fc", "http://foo.com/1/2/b/..%2fc"},
	{"http://foo.com/1/2%2f/3%2f4/5", "./a%2f../b/../c", "http://foo.com/1/2%2f/3%2f4/a%2f../c"},
	{"http://foo.com/foo%20bar/", "../baz", "http://foo.com/baz"},
	{"http://foo.com/foo", "../bar%2fbaz", "http://foo.com/bar%2fbaz"},
	{"http://foo.com/foo%2dbar/", "./baz-quux", "http://foo.com/foo%2dbar/baz-quux"},

	// RFC 3986: Normal Examples
	// http://tools.ietf.org/html/rfc3986#section-5.4.1
	{"http://a/b/c/d;p?q", "g:h", "g:h"},
	{"http://a/b/c/d;p?q", "g", "http://a/b/c/g"},
	{"http://a/b/c/d;p?q", "./g", "http://a/b/c/g"},
	{"http://a/b/c/d;p?q", "g/", "http://a/b/c/g/"},
	{"http://a/b/c/d;p?q", "/g", "http://a/g"},
	{"http://a/b/c/d;p?q", "//g", "http://g"},
	{"http://a/b/c/d;p?q", "?y", "http://a/b/c/d;p?y"},
	{"http://a/b/c/d;p?q", "g?y", "http://a/b/c/g?y"},
	{"http://a/b/c/d;p?q", "#s", "http://a/b/c/d;p?q#s"},
	{"http://a/b/c/d;p?q", "g#s", "http://a/b/c/g#s"},
	{"http://a/b/c/d;p?q", "g?y#s", "http://a/b/c/g?y#s"},
	{"http://a/b/c/d;p?q", ";x", "http://a/b/c/;x"},
	{"http://a/b/c/d;p?q", "g;x", "http://a/b/c/g;x"},
	{"http://a/b/c/d;p?q", "g;x?y#s", "http://a/b/c/g;x?y#s"},
	{"http://a/b/c/d;p?q", "", "http://a/b/c/d;p?q"},
	{"http://a/b/c/d;p?q", ".", "http://a/b/c/"},
	{"http://a/b/c/d;p?q", "./", "http://a/b/c/"},
	{"http://a/b/c/d;p?q", "..", "http://a/b/"},
	{"http://a/b/c/d;p?q", "../", "http://a/b/"},
	{"http://a/b/c/d;p?q", "../g", "http://a/b/g"},
	{"http://a/b/c/d;p?q", "../..", "http://a/"},
	{"http://a/b/c/d;p?q", "../../", "http://a/"},
	{"http://a/b/c/d;p?q", "../../g", "http://a/g"},

	// RFC 3986: Abnormal Examples
	// http://tools.ietf.org/html/rfc3986#section-5.4.2
	{"http://a/b/c/d;p?q", "../../../g", "http://a/g"},
	{"http://a/b/c/d;p?q", "../../../../g", "http://a/g"},
	{"http://a/b/c/d;p?q", "/./g", "http://a/g"},
	{"http://a/b/c/d;p?q", "/../g", "http://a/g"},
	{"http://a/b/c/d;p?q", "g.", "http://a/b/c/g."},
	{"http://a/b/c/d;p?q", ".g", "http://a/b/c/.g"},
	{"http://a/b/c/d;p?q", "g..", "http://a/b/c/g.."},
	{"http://a/b/c/d;p?q", "..g", "http://a/b/c/..g"},
	{"http://a/b/c/d;p?q", "./../g", "http://a/b/g"},
	{"http://a/b/c/d;p?q", "./g/.", "http://a/b/c/g/"},
	{"http://a/b/c/d;p?q", "g/./h", "http://a/b/c/g/h"},
	{"http://a/b/c/d;p?q", "g/../h", "http://a/b/c/h"},
	{"http://a/b/c/d;p?q", "g;x=1/./y", "http://a/b/c/g;x=1/y"},
	{"http://a/b/c/d;p?q", "g;x=1/../y", "http://a/b/c/y"},
	{"http://a/b/c/d;p?q", "g?y/./x", "http://a/b/c/g?y/./x"},
	{"http://a/b/c/d;p?q", "g?y/../x", "http://a/b/c/g?y/../x"},
	{"http://a/b/c/d;p?q", "g#s/./x", "http://a/b/c/g#s/./x"},
	{"http://a/b/c/d;p?q", "g#s/../x", "http://a/b/c/g#s/../x"},

	// Extras.
	{"https://a/b/c/d;p?q", "//g?q", "https://g?q"},
	{"https://a/b/c/d;p?q", "//g#s", "https://g#s"},
	{"https://a/b/c/d;p?q", "//g/d/e/f?y#s", "https://g/d/e/f?y#s"},
	{"https://a/b/c/d;p#s", "?y", "https://a/b/c/d;p?y"},
	{"https://a/b/c/d;p?q#s", "?y", "https://a/b/c/d;p?y"},
}

func TestResolveReference(t *testing.T) {
	mustParse := func(url string) *URL {
		u, err := Parse(url)
		if err != nil {
			t.Fatalf("Parse(%q) got err %v", url, err)
		}
		return u
	}
	opaque := &URL{Scheme: "scheme", Opaque: "opaque"}
	for _, test := range resolveReferenceTests {
		base := mustParse(test.base)
		rel := mustParse(test.rel)
		url := base.ResolveReference(rel)
		if got := url.String(); got != test.expected {
			t.Errorf("URL(%q).ResolveReference(%q)\ngot  %q\nwant %q", test.base, test.rel, got, test.expected)
		}
		// Ensure that new instances are returned.
		if base == url {
			t.Errorf("Expected URL.ResolveReference to return new URL instance.")
		}
		// Test the convenience wrapper too.
		url, err := base.Parse(test.rel)
		if err != nil {
			t.Errorf("URL(%q).Parse(%q) failed: %v", test.base, test.rel, err)
		} else if got := url.String(); got != test.expected {
			t.Errorf("URL(%q).Parse(%q)\ngot  %q\nwant %q", test.base, test.rel, got, test.expected)
		} else if base == url {
			// Ensure that new instances are returned for the wrapper too.
			t.Errorf("Expected URL.Parse to return new URL instance.")
		}
		// Ensure Opaque resets the URL.
		url = base.ResolveReference(opaque)
		if *url != *opaque {
			t.Errorf("ResolveReference failed to resolve opaque URL:\ngot  %#v\nwant %#v", url, opaque)
		}
		// Test the convenience wrapper with an opaque URL too.
		url, err = base.Parse("scheme:opaque")
		if err != nil {
			t.Errorf(`URL(%q).Parse("scheme:opaque") failed: %v`, test.base, err)
		} else if *url != *opaque {
			t.Errorf("Parse failed to resolve opaque URL:\ngot  %#v\nwant %#v", opaque, url)
		} else if base == url {
			// Ensure that new instances are returned, again.
			t.Errorf("Expected URL.Parse to return new URL instance.")
		}
	}
}

func TestQueryValues(t *testing.T) {
	u, _ := Parse("http://x.com?foo=bar&bar=1&bar=2")
	v := u.Query()
	if len(v) != 2 {
		t.Errorf("got %d keys in Query values, want 2", len(v))
	}
	if g, e := v.Get("foo"), "bar"; g != e {
		t.Errorf("Get(foo) = %q, want %q", g, e)
	}
	// Case sensitive:
	if g, e := v.Get("Foo"), ""; g != e {
		t.Errorf("Get(Foo) = %q, want %q", g, e)
	}
	if g, e := v.Get("bar"), "1"; g != e {
		t.Errorf("Get(bar) = %q, want %q", g, e)
	}
	if g, e := v.Get("baz"), ""; g != e {
		t.Errorf("Get(baz) = %q, want %q", g, e)
	}
	v.Del("bar")
	if g, e := v.Get("bar"), ""; g != e {
		t.Errorf("second Get(bar) = %q, want %q", g, e)
	}
}

type parseTest struct {
	query string
	out   Values
}

var parseTests = []parseTest{
	{
		query: "a=1&b=2",
		out:   Values{"a": []string{"1"}, "b": []string{"2"}},
	},
	{
		query: "a=1&a=2&a=banana",
		out:   Values{"a": []string{"1", "2", "banana"}},
	},
	{
		query: "ascii=%3Ckey%3A+0x90%3E",
		out:   Values{"ascii": []string{"<key: 0x90>"}},
	},
	{
		query: "a=1;b=2",
		out:   Values{"a": []string{"1"}, "b": []string{"2"}},
	},
	{
		query: "a=1&a=2;a=banana",
		out:   Values{"a": []string{"1", "2", "banana"}},
	},
}

func TestParseQuery(t *testing.T) {
	for i, test := range parseTests {
		form, err := ParseQuery(test.query)
		if err != nil {
			t.Errorf("test %d: Unexpected error: %v", i, err)
			continue
		}
		if len(form) != len(test.out) {
			t.Errorf("test %d: len(form) = %d, want %d", i, len(form), len(test.out))
		}
		for k, evs := range test.out {
			vs, ok := form[k]
			if !ok {
				t.Errorf("test %d: Missing key %q", i, k)
				continue
			}
			if len(vs) != len(evs) {
				t.Errorf("test %d: len(form[%q]) = %d, want %d", i, k, len(vs), len(evs))
				continue
			}
			for j, ev := range evs {
				if v := vs[j]; v != ev {
					t.Errorf("test %d: form[%q][%d] = %q, want %q", i, k, j, v, ev)
				}
			}
		}
	}
}

type RequestURITest struct {
	url *URL
	out string
}

var requritests = []RequestURITest{
	{
		&URL{
			Scheme: "http",
			Host:   "example.com",
			Path:   "",
		},
		"/",
	},
	{
		&URL{
			Scheme: "http",
			Host:   "example.com",
			Path:   "/a b",
		},
		"/a%20b",
	},
	// golang.org/issue/4860 variant 1
	{
		&URL{
			Scheme: "http",
			Host:   "example.com",
			Opaque: "/%2F/%2F/",
		},
		"/%2F/%2F/",
	},
	// golang.org/issue/4860 variant 2
	{
		&URL{
			Scheme: "http",
			Host:   "example.com",
			Opaque: "//other.example.com/%2F/%2F/",
		},
		"http://other.example.com/%2F/%2F/",
	},
	// better fix for issue 4860
	{
		&URL{
			Scheme:  "http",
			Host:    "example.com",
			Path:    "/////",
			RawPath: "/%2F/%2F/",
		},
		"/%2F/%2F/",
	},
	{
		&URL{
			Scheme:  "http",
			Host:    "example.com",
			Path:    "/////",
			RawPath: "/WRONG/", // ignored because doesn't match Path
		},
		"/////",
	},
	{
		&URL{
			Scheme:   "http",
			Host:     "example.com",
			Path:     "/a b",
			RawQuery: "q=go+language",
		},
		"/a%20b?q=go+language",
	},
	{
		&URL{
			Scheme:   "http",
			Host:     "example.com",
			Path:     "/a b",
			RawPath:  "/a b", // ignored because invalid
			RawQuery: "q=go+language",
		},
		"/a%20b?q=go+language",
	},
	{
		&URL{
			Scheme:   "http",
			Host:     "example.com",
			Path:     "/a?b",
			RawPath:  "/a?b", // ignored because invalid
			RawQuery: "q=go+language",
		},
		"/a%3Fb?q=go+language",
	},
	{
		&URL{
			Scheme: "myschema",
			Opaque: "opaque",
		},
		"opaque",
	},
	{
		&URL{
			Scheme:   "myschema",
			Opaque:   "opaque",
			RawQuery: "q=go+language",
		},
		"opaque?q=go+language",
	},
	{
		&URL{
			Scheme: "http",
			Host:   "example.com",
			Path:   "//foo",
		},
		"//foo",
	},
	{
		&URL{
			Scheme:     "http",
			Host:       "example.com",
			Path:       "/foo",
			ForceQuery: true,
		},
		"/foo?",
	},
}

func TestRequestURI(t *testing.T) {
	for _, tt := range requritests {
		s := tt.url.RequestURI()
		if s != tt.out {
			t.Errorf("%#v.RequestURI() == %q (expected %q)", tt.url, s, tt.out)
		}
	}
}

func TestParseFailure(t *testing.T) {
	// Test that the first parse error is returned.
	const url = "%gh&%ij"
	_, err := ParseQuery(url)
	errStr := fmt.Sprint(err)
	if !strings.Contains(errStr, "%gh") {
		t.Errorf(`ParseQuery(%q) returned error %q, want something containing %q"`, url, errStr, "%gh")
	}
}

func TestParseErrors(t *testing.T) {
	tests := []struct {
		in      string
		wantErr bool
	}{
		{"http://[::1]", false},
		{"http://[::1]:80", false},
		{"http://[::1]:namedport", true}, // rfc3986 3.2.3
		{"http://x:namedport", true},     // rfc3986 3.2.3
		{"http://[::1]/", false},
		{"http://[::1]a", true},
		{"http://[::1]%23", true},
		{"http://[::1%25en0]", false},    // valid zone id
		{"http://[::1]:", false},         // colon, but no port OK
		{"http://x:", false},             // colon, but no port OK
		{"http://[::1]:%38%30", true},    // not allowed: % encoding only for non-ASCII
		{"http://[::1%25%41]", false},    // RFC 6874 allows over-escaping in zone
		{"http://[%10::1]", true},        // no %xx escapes in IP address
		{"http://[::1]/%48", false},      // %xx in path is fine
		{"http://%41:8080/", true},       // not allowed: % encoding only for non-ASCII
		{"mysql://x@y(z:123)/foo", true}, // not well-formed per RFC 3986, golang.org/issue/33646
		{"mysql://x@y(1.2.3.4:123)/foo", true},

		{" http://foo.com", true},  // invalid character in schema
		{"ht tp://foo.com", true},  // invalid character in schema
		{"ahttp://foo.com", false}, // valid schema characters
		{"1http://foo.com", true},  // invalid character in schema

		{"http://[]%20%48%54%54%50%2f%31%2e%31%0a%4d%79%48%65%61%64%65%72%3a%20%31%32%33%0a%0a/", true}, // golang.org/issue/11208
		{"http://a b.com/", true},    // no space in host name please
		{"cache_object://foo", true}, // scheme cannot have _, relative path cannot have : in first segment
		{"cache_object:foo", true},
		{"cache_object:foo/bar", true},
		{"cache_object/:foo/bar", false},
	}
	for _, tt := range tests {
		u, err := Parse(tt.in)
		if tt.wantErr {
			if err == nil {
				t.Errorf("Parse(%q) = %#v; want an error", tt.in, u)
			}
			continue
		}
		if err != nil {
			t.Errorf("Parse(%q) = %v; want no error", tt.in, err)
		}
	}
}

// Issue 11202
func TestStarRequest(t *testing.T) {
	u, err := Parse("*")
	if err != nil {
		t.Fatal(err)
	}
	if got, want := u.RequestURI(), "*"; got != want {
		t.Errorf("RequestURI = %q; want %q", got, want)
	}
}

type shouldEscapeTest struct {
	in     byte
	mode   encoding
	escape bool
}

var shouldEscapeTests = []shouldEscapeTest{
	// Unreserved characters (§2.3)
	{'a', encodePath, false},
	{'a', encodeUserPassword, false},
	{'a', encodeQueryComponent, false},
	{'a', encodeFragment, false},
	{'a', encodeHost, false},
	{'z', encodePath, false},
	{'A', encodePath, false},
	{'Z', encodePath, false},
	{'0', encodePath, false},
	{'9', encodePath, false},
	{'-', encodePath, false},
	{'-', encodeUserPassword, false},
	{'-', encodeQueryComponent, false},
	{'-', encodeFragment, false},
	{'.', encodePath, false},
	{'_', encodePath, false},
	{'~', encodePath, false},

	// User information (§3.2.1)
	{':', encodeUserPassword, true},
	{'/', encodeUserPassword, true},
	{'?', encodeUserPassword, true},
	{'@', encodeUserPassword, true},
	{'$', encodeUserPassword, false},
	{'&', encodeUserPassword, false},
	{'+', encodeUserPassword, false},
	{',', encodeUserPassword, false},
	{';', encodeUserPassword, false},
	{'=', encodeUserPassword, false},

	// Host (IP address, IPv6 address, registered name, port suffix; §3.2.2)
	{'!', encodeHost, false},
	{'$', encodeHost, false},
	{'&', encodeHost, false},
	{'\'', encodeHost, false},
	{'(', encodeHost, false},
	{')', encodeHost, false},
	{'*', encodeHost, false},
	{'+', encodeHost, false},
	{',', encodeHost, false},
	{';', encodeHost, false},
	{'=', encodeHost, false},
	{':', encodeHost, false},
	{'[', encodeHost, false},
	{']', encodeHost, false},
	{'0', encodeHost, false},
	{'9', encodeHost, false},
	{'A', encodeHost, false},
	{'z', encodeHost, false},
	{'_', encodeHost, false},
	{'-', encodeHost, false},
	{'.', encodeHost, false},
}

func TestShouldEscape(t *testing.T) {
	for _, tt := range shouldEscapeTests {
		if shouldEscape(tt.in, tt.mode) != tt.escape {
			t.Errorf("shouldEscape(%q, %v) returned %v; expected %v", tt.in, tt.mode, !tt.escape, tt.escape)
		}
	}
}

type timeoutError struct {
	timeout bool
}

func (e *timeoutError) Error() string { return "timeout error" }
func (e *timeoutError) Timeout() bool { return e.timeout }

type temporaryError struct {
	temporary bool
}

func (e *temporaryError) Error() string   { return "temporary error" }
func (e *temporaryError) Temporary() bool { return e.temporary }

type timeoutTemporaryError struct {
	timeoutError
	temporaryError
}

func (e *timeoutTemporaryError) Error() string { return "timeout/temporary error" }

var netErrorTests = []struct {
	err       error
	timeout   bool
	temporary bool
}{{
	err:       &Error{"Get", "http://google.com/", &timeoutError{timeout: true}},
	timeout:   true,
	temporary: false,
}, {
	err:       &Error{"Get", "http://google.com/", &timeoutError{timeout: false}},
	timeout:   false,
	temporary: false,
}, {
	err:       &Error{"Get", "http://google.com/", &temporaryError{temporary: true}},
	timeout:   false,
	temporary: true,
}, {
	err:       &Error{"Get", "http://google.com/", &temporaryError{temporary: false}},
	timeout:   false,
	temporary: false,
}, {
	err:       &Error{"Get", "http://google.com/", &timeoutTemporaryError{timeoutError{timeout: true}, temporaryError{temporary: true}}},
	timeout:   true,
	temporary: true,
}, {
	err:       &Error{"Get", "http://google.com/", &timeoutTemporaryError{timeoutError{timeout: false}, temporaryError{temporary: true}}},
	timeout:   false,
	temporary: true,
}, {
	err:       &Error{"Get", "http://google.com/", &timeoutTemporaryError{timeoutError{timeout: true}, temporaryError{temporary: false}}},
	timeout:   true,
	temporary: false,
}, {
	err:       &Error{"Get", "http://google.com/", &timeoutTemporaryError{timeoutError{timeout: false}, temporaryError{temporary: false}}},
	timeout:   false,
	temporary: false,
}, {
	err:       &Error{"Get", "http://google.com/", io.EOF},
	timeout:   false,
	temporary: false,
}}

// Test that url.Error implements net.Error and that it forwards
func TestURLErrorImplementsNetError(t *testing.T) {
	for i, tt := range netErrorTests {
		err, ok := tt.err.(net.Error)
		if !ok {
			t.Errorf("%d: %T does not implement net.Error", i+1, tt.err)
			continue
		}
		if err.Timeout() != tt.timeout {
			t.Errorf("%d: err.Timeout(): got %v, want %v", i+1, err.Timeout(), tt.timeout)
			continue
		}
		if err.Temporary() != tt.temporary {
			t.Errorf("%d: err.Temporary(): got %v, want %v", i+1, err.Temporary(), tt.temporary)
		}
	}
}

func TestURLHostnameAndPort(t *testing.T) {
	tests := []struct {
		in   string // URL.Host field
		host string
		port string
	}{
		{"foo.com:80", "foo.com", "80"},
		{"foo.com", "foo.com", ""},
		{"foo.com:", "foo.com", ""},
		{"FOO.COM", "FOO.COM", ""}, // no canonicalization
		{"1.2.3.4", "1.2.3.4", ""},
		{"1.2.3.4:80", "1.2.3.4", "80"},
		{"[1:2:3:4]", "1:2:3:4", ""},
		{"[1:2:3:4]:80", "1:2:3:4", "80"},
		{"[::1]:80", "::1", "80"},
		{"[::1]", "::1", ""},
		{"[::1]:", "::1", ""},
		{"localhost", "localhost", ""},
		{"localhost:443", "localhost", "443"},
		{"some.super.long.domain.example.org:8080", "some.super.long.domain.example.org", "8080"},
		{"[2001:0db8:85a3:0000:0000:8a2e:0370:7334]:17000", "2001:0db8:85a3:0000:0000:8a2e:0370:7334", "17000"},
		{"[2001:0db8:85a3:0000:0000:8a2e:0370:7334]", "2001:0db8:85a3:0000:0000:8a2e:0370:7334", ""},

		// Ensure that even when not valid, Host is one of "Hostname",
		// "Hostname:Port", "[Hostname]" or "[Hostname]:Port".
		// See https://golang.org/issue/29098.
		{"[google.com]:80", "google.com", "80"},
		{"google.com]:80", "google.com]", "80"},
		{"google.com:80_invalid_port", "google.com:80_invalid_port", ""},
		{"[::1]extra]:80", "::1]extra", "80"},
		{"google.com]extra:extra", "google.com]extra:extra", ""},
	}
	for _, tt := range tests {
		u := &URL{Host: tt.in}
		host, port := u.Hostname(), u.Port()
		if host != tt.host {
			t.Errorf("Hostname for Host %q = %q; want %q", tt.in, host, tt.host)
		}
		if port != tt.port {
			t.Errorf("Port for Host %q = %q; want %q", tt.in, port, tt.port)
		}
	}
}

var _ encodingPkg.BinaryMarshaler = (*URL)(nil)
var _ encodingPkg.BinaryUnmarshaler = (*URL)(nil)

func TestJSON(t *testing.T) {
	u, err := Parse("https://www.google.com/x?y=z")
	if err != nil {
		t.Fatal(err)
	}
	js, err := json.Marshal(u)
	if err != nil {
		t.Fatal(err)
	}

	// If only we could implement TextMarshaler/TextUnmarshaler,
	// this would work:
	//
	// if string(js) != strconv.Quote(u.String()) {
	// 	t.Errorf("json encoding: %s\nwant: %s\n", js, strconv.Quote(u.String()))
	// }

	u1 := new(URL)
	err = json.Unmarshal(js, u1)
	if err != nil {
		t.Fatal(err)
	}
	if u1.String() != u.String() {
		t.Errorf("json decoded to: %s\nwant: %s\n", u1, u)
	}
}

func TestGob(t *testing.T) {
	u, err := Parse("https://www.google.com/x?y=z")
	if err != nil {
		t.Fatal(err)
	}
	var w bytes.Buffer
	err = gob.NewEncoder(&w).Encode(u)
	if err != nil {
		t.Fatal(err)
	}

	u1 := new(URL)
	err = gob.NewDecoder(&w).Decode(u1)
	if err != nil {
		t.Fatal(err)
	}
	if u1.String() != u.String() {
		t.Errorf("json decoded to: %s\nwant: %s\n", u1, u)
	}
}

func TestNilUser(t *testing.T) {
	defer func() {
		if v := recover(); v != nil {
			t.Fatalf("unexpected panic: %v", v)
		}
	}()

	u, err := Parse("http://foo.com/")

	if err != nil {
		t.Fatalf("parse err: %v", err)
	}

	if v := u.User.Username(); v != "" {
		t.Fatalf("expected empty username, got %s", v)
	}

	if v, ok := u.User.Password(); v != "" || ok {
		t.Fatalf("expected empty password, got %s (%v)", v, ok)
	}

	if v := u.User.String(); v != "" {
		t.Fatalf("expected empty string, got %s", v)
	}
}

func TestInvalidUserPassword(t *testing.T) {
	_, err := Parse("http://user^:passwo^rd@foo.com/")
	if got, wantsub := fmt.Sprint(err), "net/url: invalid userinfo"; !strings.Contains(got, wantsub) {
		t.Errorf("error = %q; want substring %q", got, wantsub)
	}
}

func TestRejectControlCharacters(t *testing.T) {
	tests := []string{
		"http://foo.com/?foo\nbar",
		"http\r://foo.com/",
		"http://foo\x7f.com/",
	}
	for _, s := range tests {
		_, err := Parse(s)
		const wantSub = "net/url: invalid control character in URL"
		if got := fmt.Sprint(err); !strings.Contains(got, wantSub) {
			t.Errorf("Parse(%q) error = %q; want substring %q", s, got, wantSub)
		}
	}

	// But don't reject non-ASCII CTLs, at least for now:
	if _, err := Parse("http://foo.com/ctl\x80"); err != nil {
		t.Errorf("error parsing URL with non-ASCII control byte: %v", err)
	}

}

var escapeBenchmarks = []struct {
	unescaped string
	query     string
	path      string
}{
	{
		unescaped: "one two",
		query:     "one+two",
		path:      "one%20two",
	},
	{
		unescaped: "Фотки собак",
		query:     "%D0%A4%D0%BE%D1%82%D0%BA%D0%B8+%D1%81%D0%BE%D0%B1%D0%B0%D0%BA",
		path:      "%D0%A4%D0%BE%D1%82%D0%BA%D0%B8%20%D1%81%D0%BE%D0%B1%D0%B0%D0%BA",
	},

	{
		unescaped: "shortrun(break)shortrun",
		query:     "shortrun%28break%29shortrun",
		path:      "shortrun%28break%29shortrun",
	},

	{
		unescaped: "longerrunofcharacters(break)anotherlongerrunofcharacters",
		query:     "longerrunofcharacters%28break%29anotherlongerrunofcharacters",
		path:      "longerrunofcharacters%28break%29anotherlongerrunofcharacters",
	},

	{
		unescaped: strings.Repeat("padded/with+various%characters?that=need$some@escaping+paddedsowebreak/256bytes", 4),
		query:     strings.Repeat("padded%2Fwith%2Bvarious%25characters%3Fthat%3Dneed%24some%40escaping%2Bpaddedsowebreak%2F256bytes", 4),
		path:      strings.Repeat("padded%2Fwith+various%25characters%3Fthat=need$some@escaping+paddedsowebreak%2F256bytes", 4),
	},
}

func BenchmarkQueryEscape(b *testing.B) {
	for _, tc := range escapeBenchmarks {
		b.Run("", func(b *testing.B) {
			b.ReportAllocs()
			var g string
			for i := 0; i < b.N; i++ {
				g = QueryEscape(tc.unescaped)
			}
			b.StopTimer()
			if g != tc.query {
				b.Errorf("QueryEscape(%q) == %q, want %q", tc.unescaped, g, tc.query)
			}

		})
	}
}

func BenchmarkPathEscape(b *testing.B) {
	for _, tc := range escapeBenchmarks {
		b.Run("", func(b *testing.B) {
			b.ReportAllocs()
			var g string
			for i := 0; i < b.N; i++ {
				g = PathEscape(tc.unescaped)
			}
			b.StopTimer()
			if g != tc.path {
				b.Errorf("PathEscape(%q) == %q, want %q", tc.unescaped, g, tc.path)
			}

		})
	}
}

func BenchmarkQueryUnescape(b *testing.B) {
	for _, tc := range escapeBenchmarks {
		b.Run("", func(b *testing.B) {
			b.ReportAllocs()
			var g string
			for i := 0; i < b.N; i++ {
				g, _ = QueryUnescape(tc.query)
			}
			b.StopTimer()
			if g != tc.unescaped {
				b.Errorf("QueryUnescape(%q) == %q, want %q", tc.query, g, tc.unescaped)
			}

		})
	}
}

func BenchmarkPathUnescape(b *testing.B) {
	for _, tc := range escapeBenchmarks {
		b.Run("", func(b *testing.B) {
			b.ReportAllocs()
			var g string
			for i := 0; i < b.N; i++ {
				g, _ = PathUnescape(tc.path)
			}
			b.StopTimer()
			if g != tc.unescaped {
				b.Errorf("PathUnescape(%q) == %q, want %q", tc.path, g, tc.unescaped)
			}

		})
	}
}

var sink string

func BenchmarkSplit(b *testing.B) {
	url := "http://www.google.com/?q=go+language#foo%26bar"
	for i := 0; i < b.N; i++ {
		sink, sink = split(url, '#', true)
	}
}
