// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"fmt";
	"http";
	"testing";
)

type stringMultimap map[string] []string

type parseTest struct {
	body string;
	out stringMultimap;
}

var parseTests = []parseTest{
	parseTest{
		body: "a=1&b=2",
		out: stringMultimap{ "a": []string{ "1" }, "b": []string{ "2" } },
	},
	parseTest{
		body: "a=1&a=2&a=banana",
		out: stringMultimap{ "a": []string{ "1", "2", "banana" } },
	},
	parseTest{
		body: "ascii=%3Ckey%3A+0x90%3E",
		out: stringMultimap{ "ascii": []string{ "<key: 0x90>" } },
	},
}

func TestParseForm(t *testing.T) {
	for i, test := range parseTests {
		data, err := parseForm(test.body);
		if err != nil {
			t.Errorf("test %d: Unexpected error: %v", i, err);
			continue
		}
		if dlen, olen := len(data), len(test.out); dlen != olen {
			t.Errorf("test %d: Have %d keys, want %d keys", i, dlen, olen);
		}
		for k, vs := range test.out {
			vec, ok := data[k];
			if !ok {
				t.Errorf("test %d: Missing key %q", i, k);
				continue
			}
			if dlen, olen := vec.Len(), len(vs); dlen != olen {
				t.Errorf("test %d: key %q: Have %d keys, want %d keys", i, k, dlen, olen);
				continue
			}
			for j, v := range vs {
				if dv := vec.At(j); dv != v {
					t.Errorf("test %d: key %q: val %d: Have %q, want %q", i, k, j, dv, v);
				}
			}
		}
	}
}
