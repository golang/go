// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"bytes"
	"testing"
)

func TestPrefixedInt(t *testing.T) {
	st1, st2 := newStreamPair(t)
	for _, test := range []struct {
		value     int64
		prefixLen uint8
		encoded   []byte
	}{
		// https://www.rfc-editor.org/rfc/rfc7541#appendix-C.1.1
		{
			value:     10,
			prefixLen: 5,
			encoded: []byte{
				0b_0000_1010,
			},
		},
		// https://www.rfc-editor.org/rfc/rfc7541#appendix-C.1.2
		{
			value:     1337,
			prefixLen: 5,
			encoded: []byte{
				0b0001_1111,
				0b1001_1010,
				0b0000_1010,
			},
		},
		// https://www.rfc-editor.org/rfc/rfc7541#appendix-C.1.3
		{
			value:     42,
			prefixLen: 8,
			encoded: []byte{
				0b0010_1010,
			},
		},
	} {
		highBitMask := ^((byte(1) << test.prefixLen) - 1)
		for _, highBits := range []byte{
			0, highBitMask, 0b1010_1010 & highBitMask,
		} {
			gotEnc := appendPrefixedInt(nil, highBits, test.prefixLen, test.value)
			wantEnc := append([]byte{}, test.encoded...)
			wantEnc[0] |= highBits
			if !bytes.Equal(gotEnc, wantEnc) {
				t.Errorf("appendPrefixedInt(nil, 0b%08b, %v, %v) = {%x}, want {%x}",
					highBits, test.prefixLen, test.value, gotEnc, wantEnc)
			}

			st1.Write(gotEnc)
			if err := st1.Flush(); err != nil {
				t.Fatal(err)
			}
			gotFirstByte, v, err := st2.readPrefixedInt(test.prefixLen)
			if err != nil || gotFirstByte&highBitMask != highBits || v != test.value {
				t.Errorf("st.readPrefixedInt(%v) = 0b%08b, %v, %v; want 0b%08b, %v, nil", test.prefixLen, gotFirstByte, v, err, highBits, test.value)
			}
		}
	}
}

func TestPrefixedString(t *testing.T) {
	st1, st2 := newStreamPair(t)
	for _, test := range []struct {
		value     string
		prefixLen uint8
		encoded   []byte
	}{
		// https://www.rfc-editor.org/rfc/rfc7541#appendix-C.6.1
		{
			value:     "302",
			prefixLen: 7,
			encoded: []byte{
				0x82, // H bit + length 2
				0x64, 0x02,
			},
		},
		{
			value:     "private",
			prefixLen: 5,
			encoded: []byte{
				0x25, // H bit + length 5
				0xae, 0xc3, 0x77, 0x1a, 0x4b,
			},
		},
		{
			value:     "Mon, 21 Oct 2013 20:13:21 GMT",
			prefixLen: 7,
			encoded: []byte{
				0x96, // H bit + length 22
				0xd0, 0x7a, 0xbe, 0x94, 0x10, 0x54, 0xd4, 0x44,
				0xa8, 0x20, 0x05, 0x95, 0x04, 0x0b, 0x81, 0x66,
				0xe0, 0x82, 0xa6, 0x2d, 0x1b, 0xff,
			},
		},
		{
			value:     "https://www.example.com",
			prefixLen: 7,
			encoded: []byte{
				0x91, // H bit + length 17
				0x9d, 0x29, 0xad, 0x17, 0x18, 0x63, 0xc7, 0x8f,
				0x0b, 0x97, 0xc8, 0xe9, 0xae, 0x82, 0xae, 0x43,
				0xd3,
			},
		},
		// Not Huffman encoded (encoded size == unencoded size).
		{
			value:     "a",
			prefixLen: 7,
			encoded: []byte{
				0x01, // length 1
				0x61,
			},
		},
		// Empty string.
		{
			value:     "",
			prefixLen: 7,
			encoded: []byte{
				0x00, // length 0
			},
		},
	} {
		highBitMask := ^((byte(1) << (test.prefixLen + 1)) - 1)
		for _, highBits := range []byte{
			0, highBitMask, 0b1010_1010 & highBitMask,
		} {
			gotEnc := appendPrefixedString(nil, highBits, test.prefixLen, test.value)
			wantEnc := append([]byte{}, test.encoded...)
			wantEnc[0] |= highBits
			if !bytes.Equal(gotEnc, wantEnc) {
				t.Errorf("appendPrefixedString(nil, 0b%08b, %v, %v) = {%x}, want {%x}",
					highBits, test.prefixLen, test.value, gotEnc, wantEnc)
			}

			st1.Write(gotEnc)
			if err := st1.Flush(); err != nil {
				t.Fatal(err)
			}
			gotFirstByte, v, err := st2.readPrefixedString(test.prefixLen)
			if err != nil || gotFirstByte&highBitMask != highBits || v != test.value {
				t.Errorf("st.readPrefixedInt(%v) = 0b%08b, %q, %v; want 0b%08b, %q, nil", test.prefixLen, gotFirstByte, v, err, highBits, test.value)
			}
		}
	}
}

func TestHuffmanDecodingFailure(t *testing.T) {
	st1, st2 := newStreamPair(t)
	st1.Write([]byte{
		0x82, // H bit + length 4
		0b_1111_1111,
		0b_1111_1111,
		0b_1111_1111,
		0b_1111_1111,
	})
	if err := st1.Flush(); err != nil {
		t.Fatal(err)
	}
	if b, v, err := st2.readPrefixedString(7); err == nil {
		t.Fatalf("readPrefixedString(7) = %x, %v, nil; want error", b, v)
	}
}
