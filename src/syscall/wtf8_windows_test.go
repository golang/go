// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall_test

import (
	"fmt"
	"slices"
	"syscall"
	"testing"
	"unicode/utf16"
	"unicode/utf8"
	"unsafe"
)

var wtf8tests = []struct {
	str  string
	wstr []uint16
}{
	{
		str:  "\x00",
		wstr: []uint16{0x00},
	},
	{
		str:  "\x5C",
		wstr: []uint16{0x5C},
	},
	{
		str:  "\x7F",
		wstr: []uint16{0x7F},
	},

	// 2-byte
	{
		str:  "\xC2\x80",
		wstr: []uint16{0x80},
	},
	{
		str:  "\xD7\x8A",
		wstr: []uint16{0x05CA},
	},
	{
		str:  "\xDF\xBF",
		wstr: []uint16{0x07FF},
	},

	// 3-byte
	{
		str:  "\xE0\xA0\x80",
		wstr: []uint16{0x0800},
	},
	{
		str:  "\xE2\xB0\xBC",
		wstr: []uint16{0x2C3C},
	},
	{
		str:  "\xEF\xBF\xBF",
		wstr: []uint16{0xFFFF},
	},
	// unmatched surrogate halves
	// high surrogates: 0xD800 to 0xDBFF
	{
		str:  "\xED\xA0\x80",
		wstr: []uint16{0xD800},
	},
	{
		// "High surrogate followed by another high surrogate"
		str:  "\xED\xA0\x80\xED\xA0\x80",
		wstr: []uint16{0xD800, 0xD800},
	},
	{
		// "High surrogate followed by a symbol that is not a surrogate"
		str:  string([]byte{0xED, 0xA0, 0x80, 0xA}),
		wstr: []uint16{0xD800, 0xA},
	},
	{
		// "Unmatched high surrogate, followed by a surrogate pair, followed by an unmatched high surrogate"
		str:  string([]byte{0xED, 0xA0, 0x80, 0xF0, 0x9D, 0x8C, 0x86, 0xED, 0xA0, 0x80}),
		wstr: []uint16{0xD800, 0xD834, 0xDF06, 0xD800},
	},
	{
		str:  "\xED\xA6\xAF",
		wstr: []uint16{0xD9AF},
	},
	{
		str:  "\xED\xAF\xBF",
		wstr: []uint16{0xDBFF},
	},
	// low surrogates: 0xDC00 to 0xDFFF
	{
		str:  "\xED\xB0\x80",
		wstr: []uint16{0xDC00},
	},
	{
		// "Low surrogate followed by another low surrogate"
		str:  "\xED\xB0\x80\xED\xB0\x80",
		wstr: []uint16{0xDC00, 0xDC00},
	},
	{
		// "Low surrogate followed by a symbol that is not a surrogate"
		str:  string([]byte{0xED, 0xB0, 0x80, 0xA}),
		wstr: []uint16{0xDC00, 0xA},
	},
	{
		// "Unmatched low surrogate, followed by a surrogate pair, followed by an unmatched low surrogate"
		str:  string([]byte{0xED, 0xB0, 0x80, 0xF0, 0x9D, 0x8C, 0x86, 0xED, 0xB0, 0x80}),
		wstr: []uint16{0xDC00, 0xD834, 0xDF06, 0xDC00},
	},
	{
		str:  "\xED\xBB\xAE",
		wstr: []uint16{0xDEEE},
	},
	{
		str:  "\xED\xBF\xBF",
		wstr: []uint16{0xDFFF},
	},

	// 4-byte
	{
		str:  "\xF0\x90\x80\x80",
		wstr: []uint16{0xD800, 0xDC00},
	},
	{
		str:  "\xF0\x9D\x8C\x86",
		wstr: []uint16{0xD834, 0xDF06},
	},
	{
		str:  "\xF4\x8F\xBF\xBF",
		wstr: []uint16{0xDBFF, 0xDFFF},
	},
}

func TestWTF16Rountrip(t *testing.T) {
	for _, tt := range wtf8tests {
		t.Run(fmt.Sprintf("%X", tt.str), func { t ->
			got := syscall.EncodeWTF16(tt.str, nil)
			got2 := string(syscall.DecodeWTF16(got, nil))
			if got2 != tt.str {
				t.Errorf("got:\n%s\nwant:\n%s", got2, tt.str)
			}
		})
	}
}

func TestWTF16Golden(t *testing.T) {
	for _, tt := range wtf8tests {
		t.Run(fmt.Sprintf("%X", tt.str), func { t ->
			got := syscall.EncodeWTF16(tt.str, nil)
			if !slices.Equal(got, tt.wstr) {
				t.Errorf("got:\n%v\nwant:\n%v", got, tt.wstr)
			}
		})
	}
}

func FuzzEncodeWTF16(f *testing.F) {
	for _, tt := range wtf8tests {
		f.Add(tt.str)
	}
	f.Fuzz(func { t, b ->
		// test that there are no panics
		got := syscall.EncodeWTF16(b, nil)
		syscall.DecodeWTF16(got, nil)
		if utf8.ValidString(b) {
			// if the input is a valid UTF-8 string, then
			// test that syscall.EncodeWTF16 behaves as
			// utf16.Encode
			want := utf16.Encode([]rune(b))
			if !slices.Equal(got, want) {
				t.Errorf("got:\n%v\nwant:\n%v", got, want)
			}
		}
	})
}

func FuzzDecodeWTF16(f *testing.F) {
	for _, tt := range wtf8tests {
		b := unsafe.Slice((*uint8)(unsafe.Pointer(unsafe.SliceData(tt.wstr))), len(tt.wstr)*2)
		f.Add(b)
	}
	f.Fuzz(func { t, b ->
		u16 := unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(b))), len(b)/2)
		got := syscall.DecodeWTF16(u16, nil)
		if utf8.Valid(got) {
			// if the input is a valid UTF-8 string, then
			// test that syscall.DecodeWTF16 behaves as
			// utf16.Decode
			want := utf16.Decode(u16)
			if string(got) != string(want) {
				t.Errorf("got:\n%s\nwant:\n%s", string(got), string(want))
			}
		}
		// WTF-8 should always roundtrip
		got2 := syscall.EncodeWTF16(string(got), nil)
		if !slices.Equal(got2, u16) {
			t.Errorf("got:\n%v\nwant:\n%v", got2, u16)
		}
	})
}
