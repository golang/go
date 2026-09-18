// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"bytes"
	"strings"
	"testing"
)

func TestQPACKEncode(t *testing.T) {
	type header struct {
		itype       indexType
		name, value string
	}
	// Many test cases here taken from Google QUICHE,
	// quiche/quic/core/qpack/qpack_encoder_test.cc.
	for _, test := range []struct {
		name    string
		headers []header
		want    []byte
	}{{
		name:    "empty",
		headers: []header{},
		want:    unhex("0000"),
	}, {
		name: "empty name",
		headers: []header{
			{mayIndex, "", "foo"},
		},
		want: unhex("0000208294e7"),
	}, {
		name: "empty value",
		headers: []header{
			{mayIndex, "foo", ""},
		},
		want: unhex("00002a94e700"),
	}, {
		name: "empty name and value",
		headers: []header{
			{mayIndex, "", ""},
		},
		want: unhex("00002000"),
	}, {
		name: "simple",
		headers: []header{
			{mayIndex, "foo", "bar"},
		},
		want: unhex("00002a94e703626172"),
	}, {
		name: "multiple",
		headers: []header{
			{mayIndex, "foo", "bar"},
			{mayIndex, "ZZZZZZZ", strings.Repeat("Z", 127)},
		},
		want: unhex("0000" + // prefix
			// foo: bar
			"2a94e703626172" +
			// 7 octet long header name, the smallest number
			// that does not fit on a 3-bit prefix.
			"27007a7a7a7a7a7a7a" +
			// 127 octet long header value, the smallest
			// number that does not fit on a 7-bit prefix.
			"7f005a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a" +
			"5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a" +
			"5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a" +
			"5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a"),
	}, {
		name: "static table 1",
		headers: []header{
			{mayIndex, ":method", "GET"},
			{mayIndex, "accept-encoding", "gzip, deflate, br"},
			{mayIndex, "location", ""},
		},
		want: unhex("0000d1dfcc"),
	}, {
		name: "static table 2",
		headers: []header{
			{mayIndex, ":method", "POST"},
			{mayIndex, "accept-encoding", "compress"},
			{mayIndex, "location", "foo"},
		},
		want: unhex("0000d45f108621e9aec2a11f5c8294e7"),
	}, {
		name: "static table 3",
		headers: []header{
			{mayIndex, ":method", "TRACE"},
			{mayIndex, "accept-encoding", ""},
		},
		want: unhex("00005f000554524143455f1000"),
	}, {
		name: "never indexed literal field line with name reference",
		headers: []header{
			{neverIndex, ":method", ""},
		},
		want: unhex("00007f0000"),
	}, {
		name: "never indexed literal field line with literal name",
		headers: []header{
			{neverIndex, "a", "b"},
		},
		want: unhex("000031610162"),
	}} {
		t.Run(test.name, func(t *testing.T) {
			var enc qpackEncoder
			enc.init()

			got := enc.encode(func(f func(itype indexType, name, value string)) {
				for _, h := range test.headers {
					f(h.itype, h.name, h.value)
				}
			})
			if !bytes.Equal(got, test.want) {
				for _, h := range test.headers {
					t.Logf("header %v: %q", h.name, h.value)
				}
				t.Errorf("got:  %x", got)
				t.Errorf("want: %x", test.want)
			}
		})
	}
}
