// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bytes";
	"testing";
)

type stringMultimap map[string][]string

type parseTest struct {
	query	string;
	out	stringMultimap;
}

var parseTests = []parseTest{
	parseTest{
		query: "a=1&b=2",
		out: stringMultimap{"a": []string{"1"}, "b": []string{"2"}},
	},
	parseTest{
		query: "a=1&a=2&a=banana",
		out: stringMultimap{"a": []string{"1", "2", "banana"}},
	},
	parseTest{
		query: "ascii=%3Ckey%3A+0x90%3E",
		out: stringMultimap{"ascii": []string{"<key: 0x90>"}},
	},
}

func TestParseForm(t *testing.T) {
	for i, test := range parseTests {
		form, err := parseForm(test.query);
		if err != nil {
			t.Errorf("test %d: Unexpected error: %v", i, err);
			continue;
		}
		if len(form) != len(test.out) {
			t.Errorf("test %d: len(form) = %d, want %d", i, len(form), len(test.out));
		}
		for k, evs := range test.out {
			vs, ok := form[k];
			if !ok {
				t.Errorf("test %d: Missing key %q", i, k);
				continue;
			}
			if len(vs) != len(evs) {
				t.Errorf("test %d: len(form[%q]) = %d, want %d", i, k, len(vs), len(evs));
				continue;
			}
			for j, ev := range evs {
				if v := vs[j]; v != ev {
					t.Errorf("test %d: form[%q][%d] = %q, want %q", i, k, j, v, ev);
				}
			}
		}
	}
}

func TestQuery(t *testing.T) {
	req := &Request{Method: "GET"};
	req.URL, _ = ParseURL("http://www.google.com/search?q=foo&q=bar");
	if q := req.FormValue("q"); q != "foo" {
		t.Errorf(`req.FormValue("q") = %q, want "foo"`, q);
	}
}

type stringMap map[string]string
type parseContentTypeTest struct {
	contentType	stringMap;
	error		bool;
}

var parseContentTypeTests = []parseContentTypeTest{
	parseContentTypeTest{contentType: stringMap{"Content-Type": "text/plain"}},
	parseContentTypeTest{contentType: stringMap{"Content-Type": ""}},
	parseContentTypeTest{contentType: stringMap{"Content-Type": "text/plain; boundary="}},
	parseContentTypeTest{
		contentType: stringMap{"Content-Type": "application/unknown"},
		error: true,
	},
}

func TestPostContentTypeParsing(t *testing.T) {
	for i, test := range parseContentTypeTests {
		req := &Request{
			Method: "POST",
			Header: test.contentType,
			Body: bytes.NewBufferString("body"),
		};
		err := req.ParseForm();
		if !test.error && err != nil {
			t.Errorf("test %d: Unexpected error: %v", i, err);
		}
		if test.error && err == nil {
			t.Errorf("test %d should have returned error", i);
		}
	}
}

func TestRedirect(t *testing.T) {
	const (
		start	= "http://codesearch.google.com/";
		end	= "http://www.google.com/codesearch";
	)
	r, url, err := Get(start);
	if err != nil {
		t.Fatal(err);
	}
	r.Body.Close();
	if r.StatusCode != 200 || url != end {
		t.Fatalf("Get(%s) got status %d at %s, want 200 at %s", start, r.StatusCode, url, end);
	}
}
