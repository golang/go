// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"testing"
)

type lexTest struct {
	Raw    string
	Parsed int // # of parsed characters
	Result []string
}

var lexTests = []lexTest{
	{
		Raw:    `"abc"def,:ghi`,
		Parsed: 13,
		Result: []string{"abcdef", "ghi"},
	},
	// My understanding of the RFC is that escape sequences outside of
	// quotes are not interpreted?
	{
		Raw:    `"\t"\t"\t"`,
		Parsed: 10,
		Result: []string{"\t", "t\t"},
	},
	{
		Raw:    `"\yab"\r\n`,
		Parsed: 10,
		Result: []string{"?ab", "r", "n"},
	},
	{
		Raw:    "ab\f",
		Parsed: 3,
		Result: []string{"ab?"},
	},
	{
		Raw:    "\"ab \" c,de f, gh, ij\n\t\r",
		Parsed: 23,
		Result: []string{"ab ", "c", "de", "f", "gh", "ij"},
	},
}

func min(x, y int) int {
	if x <= y {
		return x
	}
	return y
}

func TestSplitFieldValue(t *testing.T) {
	for k, l := range lexTests {
		parsed, result := httpSplitFieldValue(l.Raw)
		if parsed != l.Parsed {
			t.Errorf("#%d: Parsed %d, expected %d", k, parsed, l.Parsed)
		}
		if len(result) != len(l.Result) {
			t.Errorf("#%d: Result len  %d, expected %d", k, len(result), len(l.Result))
		}
		for i := 0; i < min(len(result), len(l.Result)); i++ {
			if result[i] != l.Result[i] {
				t.Errorf("#%d: %d-th entry mismatch. Have {%s}, expect {%s}",
					k, i, result[i], l.Result[i])
			}
		}
	}
}
