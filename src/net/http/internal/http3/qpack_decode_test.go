// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"reflect"
	"strings"
	"testing"
)

func TestQPACKDecode(t *testing.T) {
	type header struct {
		itype       indexType
		name, value string
	}
	// Many test cases here taken from Google QUICHE,
	// quiche/quic/core/qpack/qpack_encoder_test.cc.
	for _, test := range []struct {
		name string
		enc  []byte
		want []header
	}{{
		name: "empty",
		enc:  unhex("0000"),
		want: []header{},
	}, {
		name: "literal entry empty value",
		enc:  unhex("000023666f6f00"),
		want: []header{
			{mayIndex, "foo", ""},
		},
	}, {
		name: "simple literal entry",
		enc:  unhex("000023666f6f03626172"),
		want: []header{
			{mayIndex, "foo", "bar"},
		},
	}, {
		name: "multiple literal entries",
		enc: unhex("0000" + // prefix
			// foo: bar
			"23666f6f03626172" +
			// 7 octet long header name, the smallest number
			// that does not fit on a 3-bit prefix.
			"2700666f6f62616172" +
			// 127 octet long header value, the smallest number
			// that does not fit on a 7-bit prefix.
			"7f00616161616161616161616161616161616161616161616161616161616161616161" +
			"6161616161616161616161616161616161616161616161616161616161616161616161" +
			"6161616161616161616161616161616161616161616161616161616161616161616161" +
			"616161616161616161616161616161616161616161616161",
		),
		want: []header{
			{mayIndex, "foo", "bar"},
			{mayIndex, "foobaar", strings.Repeat("a", 127)},
		},
	}, {
		name: "line feed in value",
		enc:  unhex("000023666f6f0462610a72"),
		want: []header{
			{mayIndex, "foo", "ba\nr"},
		},
	}, {
		name: "huffman simple",
		enc:  unhex("00002f0125a849e95ba97d7f8925a849e95bb8e8b4bf"),
		want: []header{
			{mayIndex, "custom-key", "custom-value"},
		},
	}, {
		name: "alternating huffman nonhuffman",
		enc: unhex("0000" + // Prefix.
			"2f0125a849e95ba97d7f" + // Huffman-encoded name.
			"8925a849e95bb8e8b4bf" + // Huffman-encoded value.
			"2703637573746f6d2d6b6579" + // Non-Huffman encoded name.
			"0c637573746f6d2d76616c7565" + // Non-Huffman encoded value.
			"2f0125a849e95ba97d7f" + // Huffman-encoded name.
			"0c637573746f6d2d76616c7565" + // Non-Huffman encoded value.
			"2703637573746f6d2d6b6579" + // Non-Huffman encoded name.
			"8925a849e95bb8e8b4bf", // Huffman-encoded value.
		),
		want: []header{
			{mayIndex, "custom-key", "custom-value"},
			{mayIndex, "custom-key", "custom-value"},
			{mayIndex, "custom-key", "custom-value"},
			{mayIndex, "custom-key", "custom-value"},
		},
	}, {
		name: "static table",
		enc:  unhex("0000d1d45f00055452414345dfcc5f108621e9aec2a11f5c8294e75f1000"),
		want: []header{
			{mayIndex, ":method", "GET"},
			{mayIndex, ":method", "POST"},
			{mayIndex, ":method", "TRACE"},
			{mayIndex, "accept-encoding", "gzip, deflate, br"},
			{mayIndex, "location", ""},
			{mayIndex, "accept-encoding", "compress"},
			{mayIndex, "location", "foo"},
			{mayIndex, "accept-encoding", ""},
		},
	}} {
		synctestSubtest(t, test.name, func(t *testing.T) {
			st1, st2 := newStreamPair(t)
			st1.Write(test.enc)
			st1.Flush()

			st2.lim = int64(len(test.enc))

			var dec qpackDecoder
			got := []header{}
			err := dec.decode(st2, func(itype indexType, name, value string) error {
				got = append(got, header{itype, name, value})
				return nil
			})
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			if !reflect.DeepEqual(got, test.want) {
				t.Errorf("encoded: %x", test.enc)
				t.Errorf("got headers:")
				for _, h := range got {
					t.Errorf("  %v: %q", h.name, h.value)
				}
				t.Errorf("want headers:")
				for _, h := range test.want {
					t.Errorf("  %v: %q", h.name, h.value)
				}
			}
		})
	}
}

func TestQPACKDecodeErrors(t *testing.T) {
	// Many test cases here taken from Google QUICHE,
	// quiche/quic/core/qpack/qpack_encoder_test.cc.
	for _, test := range []struct {
		name string
		enc  []byte
	}{{
		name: "literal entry empty name",
		enc:  unhex("00002003666f6f"),
	}, {
		name: "literal entry empty name and value",
		enc:  unhex("00002000"),
	}, {
		name: "name length too large for varint",
		enc:  unhex("000027ffffffffffffffffffff"),
	}, {
		name: "string literal too long",
		enc:  unhex("000027ffff7f"),
	}, {
		name: "value length too large for varint",
		enc:  unhex("000023666f6f7fffffffffffffffffffff"),
	}, {
		name: "value length too long",
		enc:  unhex("000023666f6f7fffff7f"),
	}, {
		name: "incomplete header block",
		enc:  unhex("00002366"),
	}, {
		name: "huffman name does not have eos prefix",
		enc:  unhex("00002f0125a849e95ba97d7e8925a849e95bb8e8b4bf"),
	}, {
		name: "huffman value does not have eos prefix",
		enc:  unhex("00002f0125a849e95ba97d7f8925a849e95bb8e8b4be"),
	}, {
		name: "huffman name eos prefix too long",
		enc:  unhex("00002f0225a849e95ba97d7fff8925a849e95bb8e8b4bf"),
	}, {
		name: "huffman value eos prefix too long",
		enc:  unhex("00002f0125a849e95ba97d7f8a25a849e95bb8e8b4bfff"),
	}, {
		name: "too high static table index",
		enc:  unhex("0000ff23ff24"),
	}, {
		name: "prefixed string length overflow",
		enc:  unhex("000027ffffffffffffffff7f"),
	}, {
		name: "prefixed static index overflow",
		enc:  unhex("0000ffffffffffffffffff7f"),
	}} {
		synctestSubtest(t, test.name, func(t *testing.T) {
			st1, st2 := newStreamPair(t)
			st1.Write(test.enc)
			st1.Flush()

			st2.lim = int64(len(test.enc))

			var dec qpackDecoder
			err := dec.decode(st2, func(itype indexType, name, value string) error {
				return nil
			})
			if err == nil {
				t.Errorf("encoded: %x", test.enc)
				t.Fatalf("decode succeeded; want error")
			}
		})
	}
}
