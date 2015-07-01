// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

var writeSetCookiesTests = []struct {
	Cookie *Cookie
	Raw    string
}{
	{
		&Cookie{Name: "cookie-1", Value: "v$1"},
		"cookie-1=v$1",
	},
	{
		&Cookie{Name: "cookie-2", Value: "two", MaxAge: 3600},
		"cookie-2=two; Max-Age=3600",
	},
	{
		&Cookie{Name: "cookie-3", Value: "three", Domain: ".example.com"},
		"cookie-3=three; Domain=example.com",
	},
	{
		&Cookie{Name: "cookie-4", Value: "four", Path: "/restricted/"},
		"cookie-4=four; Path=/restricted/",
	},
	{
		&Cookie{Name: "cookie-5", Value: "five", Domain: "wrong;bad.abc"},
		"cookie-5=five",
	},
	{
		&Cookie{Name: "cookie-6", Value: "six", Domain: "bad-.abc"},
		"cookie-6=six",
	},
	{
		&Cookie{Name: "cookie-7", Value: "seven", Domain: "127.0.0.1"},
		"cookie-7=seven; Domain=127.0.0.1",
	},
	{
		&Cookie{Name: "cookie-8", Value: "eight", Domain: "::1"},
		"cookie-8=eight",
	},
	{
		&Cookie{Name: "cookie-9", Value: "expiring", Expires: time.Unix(1257894000, 0)},
		"cookie-9=expiring; Expires=Tue, 10 Nov 2009 23:00:00 GMT",
	},
	// The "special" cookies have values containing commas or spaces which
	// are disallowed by RFC 6265 but are common in the wild.
	{
		&Cookie{Name: "special-1", Value: "a z"},
		`special-1=a z`,
	},
	{
		&Cookie{Name: "special-2", Value: " z"},
		`special-2=" z"`,
	},
	{
		&Cookie{Name: "special-3", Value: "a "},
		`special-3="a "`,
	},
	{
		&Cookie{Name: "special-4", Value: " "},
		`special-4=" "`,
	},
	{
		&Cookie{Name: "special-5", Value: "a,z"},
		`special-5=a,z`,
	},
	{
		&Cookie{Name: "special-6", Value: ",z"},
		`special-6=",z"`,
	},
	{
		&Cookie{Name: "special-7", Value: "a,"},
		`special-7="a,"`,
	},
	{
		&Cookie{Name: "special-8", Value: ","},
		`special-8=","`,
	},
	{
		&Cookie{Name: "empty-value", Value: ""},
		`empty-value=`,
	},
	{
		nil,
		``,
	},
	{
		&Cookie{Name: ""},
		``,
	},
	{
		&Cookie{Name: "\t"},
		``,
	},
}

func TestWriteSetCookies(t *testing.T) {
	defer log.SetOutput(os.Stderr)
	var logbuf bytes.Buffer
	log.SetOutput(&logbuf)

	for i, tt := range writeSetCookiesTests {
		if g, e := tt.Cookie.String(), tt.Raw; g != e {
			t.Errorf("Test %d, expecting:\n%s\nGot:\n%s\n", i, e, g)
			continue
		}
	}

	if got, sub := logbuf.String(), "dropping domain attribute"; !strings.Contains(got, sub) {
		t.Errorf("Expected substring %q in log output. Got:\n%s", sub, got)
	}
}

type headerOnlyResponseWriter Header

func (ho headerOnlyResponseWriter) Header() Header {
	return Header(ho)
}

func (ho headerOnlyResponseWriter) Write([]byte) (int, error) {
	panic("NOIMPL")
}

func (ho headerOnlyResponseWriter) WriteHeader(int) {
	panic("NOIMPL")
}

func TestSetCookie(t *testing.T) {
	m := make(Header)
	SetCookie(headerOnlyResponseWriter(m), &Cookie{Name: "cookie-1", Value: "one", Path: "/restricted/"})
	SetCookie(headerOnlyResponseWriter(m), &Cookie{Name: "cookie-2", Value: "two", MaxAge: 3600})
	if l := len(m["Set-Cookie"]); l != 2 {
		t.Fatalf("expected %d cookies, got %d", 2, l)
	}
	if g, e := m["Set-Cookie"][0], "cookie-1=one; Path=/restricted/"; g != e {
		t.Errorf("cookie #1: want %q, got %q", e, g)
	}
	if g, e := m["Set-Cookie"][1], "cookie-2=two; Max-Age=3600"; g != e {
		t.Errorf("cookie #2: want %q, got %q", e, g)
	}
}

var addCookieTests = []struct {
	Cookies []*Cookie
	Raw     string
}{
	{
		[]*Cookie{},
		"",
	},
	{
		[]*Cookie{{Name: "cookie-1", Value: "v$1"}},
		"cookie-1=v$1",
	},
	{
		[]*Cookie{
			{Name: "cookie-1", Value: "v$1"},
			{Name: "cookie-2", Value: "v$2"},
			{Name: "cookie-3", Value: "v$3"},
		},
		"cookie-1=v$1; cookie-2=v$2; cookie-3=v$3",
	},
}

func TestAddCookie(t *testing.T) {
	for i, tt := range addCookieTests {
		req, _ := NewRequest("GET", "http://example.com/", nil)
		for _, c := range tt.Cookies {
			req.AddCookie(c)
		}
		if g := req.Header.Get("Cookie"); g != tt.Raw {
			t.Errorf("Test %d:\nwant: %s\n got: %s\n", i, tt.Raw, g)
			continue
		}
	}
}

var readSetCookiesTests = []struct {
	Header  Header
	Cookies []*Cookie
}{
	{
		Header{"Set-Cookie": {"Cookie-1=v$1"}},
		[]*Cookie{{Name: "Cookie-1", Value: "v$1", Raw: "Cookie-1=v$1"}},
	},
	{
		Header{"Set-Cookie": {"NID=99=YsDT5i3E-CXax-; expires=Wed, 23-Nov-2011 01:05:03 GMT; path=/; domain=.google.ch; HttpOnly"}},
		[]*Cookie{{
			Name:       "NID",
			Value:      "99=YsDT5i3E-CXax-",
			Path:       "/",
			Domain:     ".google.ch",
			HttpOnly:   true,
			Expires:    time.Date(2011, 11, 23, 1, 5, 3, 0, time.UTC),
			RawExpires: "Wed, 23-Nov-2011 01:05:03 GMT",
			Raw:        "NID=99=YsDT5i3E-CXax-; expires=Wed, 23-Nov-2011 01:05:03 GMT; path=/; domain=.google.ch; HttpOnly",
		}},
	},
	{
		Header{"Set-Cookie": {".ASPXAUTH=7E3AA; expires=Wed, 07-Mar-2012 14:25:06 GMT; path=/; HttpOnly"}},
		[]*Cookie{{
			Name:       ".ASPXAUTH",
			Value:      "7E3AA",
			Path:       "/",
			Expires:    time.Date(2012, 3, 7, 14, 25, 6, 0, time.UTC),
			RawExpires: "Wed, 07-Mar-2012 14:25:06 GMT",
			HttpOnly:   true,
			Raw:        ".ASPXAUTH=7E3AA; expires=Wed, 07-Mar-2012 14:25:06 GMT; path=/; HttpOnly",
		}},
	},
	{
		Header{"Set-Cookie": {"ASP.NET_SessionId=foo; path=/; HttpOnly"}},
		[]*Cookie{{
			Name:     "ASP.NET_SessionId",
			Value:    "foo",
			Path:     "/",
			HttpOnly: true,
			Raw:      "ASP.NET_SessionId=foo; path=/; HttpOnly",
		}},
	},
	// Make sure we can properly read back the Set-Cookie headers we create
	// for values containing spaces or commas:
	{
		Header{"Set-Cookie": {`special-1=a z`}},
		[]*Cookie{{Name: "special-1", Value: "a z", Raw: `special-1=a z`}},
	},
	{
		Header{"Set-Cookie": {`special-2=" z"`}},
		[]*Cookie{{Name: "special-2", Value: " z", Raw: `special-2=" z"`}},
	},
	{
		Header{"Set-Cookie": {`special-3="a "`}},
		[]*Cookie{{Name: "special-3", Value: "a ", Raw: `special-3="a "`}},
	},
	{
		Header{"Set-Cookie": {`special-4=" "`}},
		[]*Cookie{{Name: "special-4", Value: " ", Raw: `special-4=" "`}},
	},
	{
		Header{"Set-Cookie": {`special-5=a,z`}},
		[]*Cookie{{Name: "special-5", Value: "a,z", Raw: `special-5=a,z`}},
	},
	{
		Header{"Set-Cookie": {`special-6=",z"`}},
		[]*Cookie{{Name: "special-6", Value: ",z", Raw: `special-6=",z"`}},
	},
	{
		Header{"Set-Cookie": {`special-7=a,`}},
		[]*Cookie{{Name: "special-7", Value: "a,", Raw: `special-7=a,`}},
	},
	{
		Header{"Set-Cookie": {`special-8=","`}},
		[]*Cookie{{Name: "special-8", Value: ",", Raw: `special-8=","`}},
	},

	// TODO(bradfitz): users have reported seeing this in the
	// wild, but do browsers handle it? RFC 6265 just says "don't
	// do that" (section 3) and then never mentions header folding
	// again.
	// Header{"Set-Cookie": {"ASP.NET_SessionId=foo; path=/; HttpOnly, .ASPXAUTH=7E3AA; expires=Wed, 07-Mar-2012 14:25:06 GMT; path=/; HttpOnly"}},
}

func toJSON(v interface{}) string {
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf("%#v", v)
	}
	return string(b)
}

func TestReadSetCookies(t *testing.T) {
	for i, tt := range readSetCookiesTests {
		for n := 0; n < 2; n++ { // to verify readSetCookies doesn't mutate its input
			c := readSetCookies(tt.Header)
			if !reflect.DeepEqual(c, tt.Cookies) {
				t.Errorf("#%d readSetCookies: have\n%s\nwant\n%s\n", i, toJSON(c), toJSON(tt.Cookies))
				continue
			}
		}
	}
}

var readCookiesTests = []struct {
	Header  Header
	Filter  string
	Cookies []*Cookie
}{
	{
		Header{"Cookie": {"Cookie-1=v$1", "c2=v2"}},
		"",
		[]*Cookie{
			{Name: "Cookie-1", Value: "v$1"},
			{Name: "c2", Value: "v2"},
		},
	},
	{
		Header{"Cookie": {"Cookie-1=v$1", "c2=v2"}},
		"c2",
		[]*Cookie{
			{Name: "c2", Value: "v2"},
		},
	},
	{
		Header{"Cookie": {"Cookie-1=v$1; c2=v2"}},
		"",
		[]*Cookie{
			{Name: "Cookie-1", Value: "v$1"},
			{Name: "c2", Value: "v2"},
		},
	},
	{
		Header{"Cookie": {"Cookie-1=v$1; c2=v2"}},
		"c2",
		[]*Cookie{
			{Name: "c2", Value: "v2"},
		},
	},
	{
		Header{"Cookie": {`Cookie-1="v$1"; c2="v2"`}},
		"",
		[]*Cookie{
			{Name: "Cookie-1", Value: "v$1"},
			{Name: "c2", Value: "v2"},
		},
	},
}

func TestReadCookies(t *testing.T) {
	for i, tt := range readCookiesTests {
		for n := 0; n < 2; n++ { // to verify readCookies doesn't mutate its input
			c := readCookies(tt.Header, tt.Filter)
			if !reflect.DeepEqual(c, tt.Cookies) {
				t.Errorf("#%d readCookies:\nhave: %s\nwant: %s\n", i, toJSON(c), toJSON(tt.Cookies))
				continue
			}
		}
	}
}

func TestSetCookieDoubleQuotes(t *testing.T) {
	res := &Response{Header: Header{}}
	res.Header.Add("Set-Cookie", `quoted0=none; max-age=30`)
	res.Header.Add("Set-Cookie", `quoted1="cookieValue"; max-age=31`)
	res.Header.Add("Set-Cookie", `quoted2=cookieAV; max-age="32"`)
	res.Header.Add("Set-Cookie", `quoted3="both"; max-age="33"`)
	got := res.Cookies()
	want := []*Cookie{
		{Name: "quoted0", Value: "none", MaxAge: 30},
		{Name: "quoted1", Value: "cookieValue", MaxAge: 31},
		{Name: "quoted2", Value: "cookieAV"},
		{Name: "quoted3", Value: "both"},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d cookies, want %d", len(got), len(want))
	}
	for i, w := range want {
		g := got[i]
		if g.Name != w.Name || g.Value != w.Value || g.MaxAge != w.MaxAge {
			t.Errorf("cookie #%d:\ngot  %v\nwant %v", i, g, w)
		}
	}
}

func TestCookieSanitizeValue(t *testing.T) {
	defer log.SetOutput(os.Stderr)
	var logbuf bytes.Buffer
	log.SetOutput(&logbuf)

	tests := []struct {
		in, want string
	}{
		{"foo", "foo"},
		{"foo;bar", "foobar"},
		{"foo\\bar", "foobar"},
		{"foo\"bar", "foobar"},
		{"\x00\x7e\x7f\x80", "\x7e"},
		{`"withquotes"`, "withquotes"},
		{"a z", "a z"},
		{" z", `" z"`},
		{"a ", `"a "`},
	}
	for _, tt := range tests {
		if got := sanitizeCookieValue(tt.in); got != tt.want {
			t.Errorf("sanitizeCookieValue(%q) = %q; want %q", tt.in, got, tt.want)
		}
	}

	if got, sub := logbuf.String(), "dropping invalid bytes"; !strings.Contains(got, sub) {
		t.Errorf("Expected substring %q in log output. Got:\n%s", sub, got)
	}
}

func TestCookieSanitizePath(t *testing.T) {
	defer log.SetOutput(os.Stderr)
	var logbuf bytes.Buffer
	log.SetOutput(&logbuf)

	tests := []struct {
		in, want string
	}{
		{"/path", "/path"},
		{"/path with space/", "/path with space/"},
		{"/just;no;semicolon\x00orstuff/", "/justnosemicolonorstuff/"},
	}
	for _, tt := range tests {
		if got := sanitizeCookiePath(tt.in); got != tt.want {
			t.Errorf("sanitizeCookiePath(%q) = %q; want %q", tt.in, got, tt.want)
		}
	}

	if got, sub := logbuf.String(), "dropping invalid bytes"; !strings.Contains(got, sub) {
		t.Errorf("Expected substring %q in log output. Got:\n%s", sub, got)
	}
}
