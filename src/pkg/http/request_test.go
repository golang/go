// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"bytes"
	"fmt"
	. "http"
	"http/httptest"
	"io"
	"os"
	"reflect"
	"regexp"
	"strings"
	"testing"
)

type stringMultimap map[string][]string

type parseTest struct {
	query string
	out   stringMultimap
}

var parseTests = []parseTest{
	{
		query: "a=1&b=2",
		out:   stringMultimap{"a": []string{"1"}, "b": []string{"2"}},
	},
	{
		query: "a=1&a=2&a=banana",
		out:   stringMultimap{"a": []string{"1", "2", "banana"}},
	},
	{
		query: "ascii=%3Ckey%3A+0x90%3E",
		out:   stringMultimap{"ascii": []string{"<key: 0x90>"}},
	},
}

func TestParseForm(t *testing.T) {
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

func TestQuery(t *testing.T) {
	req := &Request{Method: "GET"}
	req.URL, _ = ParseURL("http://www.google.com/search?q=foo&q=bar")
	if q := req.FormValue("q"); q != "foo" {
		t.Errorf(`req.FormValue("q") = %q, want "foo"`, q)
	}
}

func TestPostQuery(t *testing.T) {
	req := &Request{Method: "POST"}
	req.URL, _ = ParseURL("http://www.google.com/search?q=foo&q=bar&both=x")
	req.Header = Header{
		"Content-Type": {"application/x-www-form-urlencoded; boo!"},
	}
	req.Body = nopCloser{strings.NewReader("z=post&both=y")}
	if q := req.FormValue("q"); q != "foo" {
		t.Errorf(`req.FormValue("q") = %q, want "foo"`, q)
	}
	if z := req.FormValue("z"); z != "post" {
		t.Errorf(`req.FormValue("z") = %q, want "post"`, z)
	}
	if both := req.Form["both"]; !reflect.DeepEqual(both, []string{"x", "y"}) {
		t.Errorf(`req.FormValue("both") = %q, want ["x", "y"]`, both)
	}
}

type stringMap map[string][]string
type parseContentTypeTest struct {
	contentType stringMap
	error       bool
}

var parseContentTypeTests = []parseContentTypeTest{
	{contentType: stringMap{"Content-Type": {"text/plain"}}},
	{contentType: stringMap{}}, // Non-existent keys are not placed. The value nil is illegal.
	{contentType: stringMap{"Content-Type": {"text/plain; boundary="}}},
	{
		contentType: stringMap{"Content-Type": {"application/unknown"}},
		error:       true,
	},
}

func TestPostContentTypeParsing(t *testing.T) {
	for i, test := range parseContentTypeTests {
		req := &Request{
			Method: "POST",
			Header: Header(test.contentType),
			Body:   nopCloser{bytes.NewBufferString("body")},
		}
		err := req.ParseForm()
		if !test.error && err != nil {
			t.Errorf("test %d: Unexpected error: %v", i, err)
		}
		if test.error && err == nil {
			t.Errorf("test %d should have returned error", i)
		}
	}
}

func TestMultipartReader(t *testing.T) {
	req := &Request{
		Method: "POST",
		Header: Header{"Content-Type": {`multipart/form-data; boundary="foo123"`}},
		Body:   nopCloser{new(bytes.Buffer)},
	}
	multipart, err := req.MultipartReader()
	if multipart == nil {
		t.Errorf("expected multipart; error: %v", err)
	}

	req.Header = Header{"Content-Type": {"text/plain"}}
	multipart, err = req.MultipartReader()
	if multipart != nil {
		t.Errorf("unexpected multipart for text/plain")
	}
}

func TestRedirect(t *testing.T) {
	ts := httptest.NewServer(HandlerFunc(func(w ResponseWriter, r *Request) {
		switch r.URL.Path {
		case "/":
			w.Header().Set("Location", "/foo/")
			w.WriteHeader(StatusSeeOther)
		case "/foo/":
			fmt.Fprintf(w, "foo")
		default:
			w.WriteHeader(StatusBadRequest)
		}
	}))
	defer ts.Close()

	var end = regexp.MustCompile("/foo/$")
	r, url, err := Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	if r.StatusCode != 200 || !end.MatchString(url) {
		t.Fatalf("Get got status %d at %q, want 200 matching /foo/$", r.StatusCode, url)
	}
}

// TODO: stop copy/pasting this around.  move to io/ioutil?
type nopCloser struct {
	io.Reader
}

func (nopCloser) Close() os.Error { return nil }
