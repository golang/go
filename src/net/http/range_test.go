// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"runtime/metrics"
	"strings"
	"testing"
)

var ParseRangeTests = []struct {
	s      string
	length int64
	r      []httpRange
}{
	{"", 0, nil},
	{"", 1000, nil},
	{"foo", 0, nil},
	{"bytes=", 0, nil},
	{"bytes=7", 10, nil},
	{"bytes= 7 ", 10, nil},
	{"bytes=1-", 0, nil},
	{"bytes=5-4", 10, nil},
	{"bytes=0-2,5-4", 10, nil},
	{"bytes=2-5,4-3", 10, nil},
	{"bytes=--5,4--3", 10, nil},
	{"bytes=A-", 10, nil},
	{"bytes=A- ", 10, nil},
	{"bytes=A-Z", 10, nil},
	{"bytes= -Z", 10, nil},
	{"bytes=5-Z", 10, nil},
	{"bytes=Ran-dom, garbage", 10, nil},
	{"bytes=0x01-0x02", 10, nil},
	{"bytes=         ", 10, nil},
	{"bytes= , , ,   ", 10, nil},

	{"bytes=0-9", 10, []httpRange{{0, 10}}},
	{"Bytes=0-9", 10, []httpRange{{0, 10}}},
	{"BYTES=0-9", 10, []httpRange{{0, 10}}},
	{"bYtEs=0-9", 10, []httpRange{{0, 10}}},
	{"bytes=0-", 10, []httpRange{{0, 10}}},
	{"bytes=5-", 10, []httpRange{{5, 5}}},
	{"bytes=0-20", 10, []httpRange{{0, 10}}},
	{"bytes=15-,0-5", 10, []httpRange{{0, 6}}},
	{"bytes=1-2,5-", 10, []httpRange{{1, 2}, {5, 5}}},
	{"bytes=-2 , 7-", 11, []httpRange{{9, 2}, {7, 4}}},
	{"bytes=0-0 ,2-2, 7-", 11, []httpRange{{0, 1}, {2, 1}, {7, 4}}},
	{"bytes=-5", 10, []httpRange{{5, 5}}},
	{"bytes=-15", 10, []httpRange{{0, 10}}},
	{"bytes=0-499", 10000, []httpRange{{0, 500}}},
	{"bytes=500-999", 10000, []httpRange{{500, 500}}},
	{"bytes=-500", 10000, []httpRange{{9500, 500}}},
	{"bytes=9500-", 10000, []httpRange{{9500, 500}}},
	{"bytes=0-0,-1", 10000, []httpRange{{0, 1}, {9999, 1}}},
	{"bytes=500-600,601-999", 10000, []httpRange{{500, 101}, {601, 399}}},
	{"bytes=500-700,601-999", 10000, []httpRange{{500, 201}, {601, 399}}},

	// Match Apache laxity:
	{"bytes=   1 -2   ,  4- 5, 7 - 8 , ,,", 11, []httpRange{{1, 2}, {4, 2}, {7, 2}}},
}

func TestParseRange(t *testing.T) {
	for _, test := range ParseRangeTests {
		r := test.r
		ranges, err := parseRange(test.s, test.length)
		if err != nil && r != nil {
			t.Errorf("parseRange(%q) returned error %q", test.s, err)
		}
		if len(ranges) != len(r) {
			t.Errorf("len(parseRange(%q)) = %d, want %d", test.s, len(ranges), len(r))
			continue
		}
		for i := range r {
			if ranges[i].start != r[i].start {
				t.Errorf("parseRange(%q)[%d].start = %d, want %d", test.s, i, ranges[i].start, r[i].start)
			}
			if ranges[i].length != r[i].length {
				t.Errorf("parseRange(%q)[%d].length = %d, want %d", test.s, i, ranges[i].length, r[i].length)
			}
		}
	}
}

func TestParseRangeLimit(t *testing.T) {
	for _, tc := range []struct {
		name       string
		godebug    string
		numRanges  int
		want       int
		wantNonDef bool
	}{
		{
			name:      "default limit not exceeded",
			numRanges: defaultMaxContentRanges,
			want:      defaultMaxContentRanges,
		},
		{
			name:      "default limit exceeded",
			numRanges: defaultMaxContentRanges + 1,
			want:      0,
		},
		{
			name:       "small limit exceeded",
			godebug:    "httpservecontentmaxranges=10",
			numRanges:  11,
			want:       0,
			wantNonDef: true,
		},
		{
			name:      "small limit not exceeded",
			godebug:   "httpservecontentmaxranges=10",
			numRanges: 10,
			want:      10,
		},
		{
			name:       "disabled limit",
			godebug:    "httpservecontentmaxranges=0",
			numRanges:  2 * defaultMaxContentRanges,
			want:       2 * defaultMaxContentRanges,
			wantNonDef: true,
		},
		{
			name:      "large limit exceeded",
			godebug:   "httpservecontentmaxranges=300",
			numRanges: 301,
			want:      0,
		},
		{
			name:       "large limit not exceeded",
			godebug:    "httpservecontentmaxranges=300",
			numRanges:  300,
			want:       300,
			wantNonDef: true,
		},
		{
			name:      "large limit not exceeded by small input",
			godebug:   "httpservecontentmaxranges=300",
			numRanges: 10,
			want:      10,
		},
		{
			name:      "invalid limit negative",
			godebug:   "httpservecontentmaxranges=-5",
			numRanges: defaultMaxContentRanges + 1,
			want:      0,
		},
		{
			name:      "invalid limit non-numeric",
			godebug:   "httpservecontentmaxranges=abc",
			numRanges: defaultMaxContentRanges + 1,
			want:      0,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("GODEBUG", tc.godebug)
			var m [1]metrics.Sample
			m[0].Name = "/godebug/non-default-behavior/httpservecontentmaxranges:events"
			metrics.Read(m[:])
			before := m[0].Value.Uint64()

			rangeHeader := "bytes=" + strings.Repeat("0-0,", tc.numRanges-1) + "0-0"
			ranges, err := parseRange(rangeHeader, 10_000_000)
			if err != nil {
				t.Fatalf("parseRange(%q): %v", rangeHeader, err)
			}
			if got := len(ranges); got != tc.want {
				t.Errorf("len(ranges) = %v, want %v", got, tc.want)
			}

			metrics.Read(m[:])
			after := m[0].Value.Uint64()
			if tc.wantNonDef && after <= before {
				t.Errorf("metric did not increment: before=%d, after=%d", before, after)
			} else if !tc.wantNonDef && after != before {
				t.Errorf("metric unexpectedly incremented: before=%d, after=%d", before, after)
			}
		})
	}
}
