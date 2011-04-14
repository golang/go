// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bytes"
	"fmt"
	"json"
	"reflect"
	"testing"
)


var writeSetCookiesTests = []struct {
	Cookies []*Cookie
	Raw     string
}{
	{
		[]*Cookie{
			&Cookie{Name: "cookie-1", Value: "v$1"},
			&Cookie{Name: "cookie-2", Value: "two", MaxAge: 3600},
			&Cookie{Name: "cookie-3", Value: "three", Domain: ".example.com"},
			&Cookie{Name: "cookie-4", Value: "four", Path: "/restricted/"},
		},
		"Set-Cookie: cookie-1=v$1\r\n" +
			"Set-Cookie: cookie-2=two; Max-Age=3600\r\n" +
			"Set-Cookie: cookie-3=three; Domain=.example.com\r\n" +
			"Set-Cookie: cookie-4=four; Path=/restricted/\r\n",
	},
}

func TestWriteSetCookies(t *testing.T) {
	for i, tt := range writeSetCookiesTests {
		var w bytes.Buffer
		writeSetCookies(&w, tt.Cookies)
		seen := string(w.Bytes())
		if seen != tt.Raw {
			t.Errorf("Test %d, expecting:\n%s\nGot:\n%s\n", i, tt.Raw, seen)
			continue
		}
	}
}

var writeCookiesTests = []struct {
	Cookies []*Cookie
	Raw     string
}{
	{
		[]*Cookie{&Cookie{Name: "cookie-1", Value: "v$1"}},
		"Cookie: cookie-1=v$1\r\n",
	},
}

func TestWriteCookies(t *testing.T) {
	for i, tt := range writeCookiesTests {
		var w bytes.Buffer
		writeCookies(&w, tt.Cookies)
		seen := string(w.Bytes())
		if seen != tt.Raw {
			t.Errorf("Test %d, expecting:\n%s\nGot:\n%s\n", i, tt.Raw, seen)
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
		[]*Cookie{&Cookie{Name: "Cookie-1", Value: "v$1", Raw: "Cookie-1=v$1"}},
	},
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
		c := readSetCookies(tt.Header)
		if !reflect.DeepEqual(c, tt.Cookies) {
			t.Errorf("#%d readSetCookies: have\n%s\nwant\n%s\n", i, toJSON(c), toJSON(tt.Cookies))
			continue
		}
	}
}

var readCookiesTests = []struct {
	Header  Header
	Cookies []*Cookie
}{
	{
		Header{"Cookie": {"Cookie-1=v$1"}},
		[]*Cookie{&Cookie{Name: "Cookie-1", Value: "v$1"}},
	},
}

func TestReadCookies(t *testing.T) {
	for i, tt := range readCookiesTests {
		c := readCookies(tt.Header)
		if !reflect.DeepEqual(c, tt.Cookies) {
			t.Errorf("#%d readCookies: have\n%s\nwant\n%s\n", i, toJSON(c), toJSON(tt.Cookies))
			continue
		}
	}
}
