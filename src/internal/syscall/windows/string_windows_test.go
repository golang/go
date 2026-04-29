// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows_test

import (
	"internal/syscall/windows"
	"math"
	"strings"
	"syscall"
	"testing"
	"unsafe"
)

func TestRoundtripNTUnicodeString(t *testing.T) {
	// NTUnicodeString maximum string length must fit in a uint16, less for terminal NUL.
	maxString := strings.Repeat("*", (math.MaxUint16/2)-1)
	for _, test := range []struct {
		s       string
		wantErr bool
	}{{
		s: "",
	}, {
		s: "hello",
	}, {
		s: "Ƀ",
	}, {
		s: maxString,
	}, {
		s:       maxString + "*",
		wantErr: true,
	}, {
		s:       "a\x00a",
		wantErr: true,
	}} {
		ntus, err := windows.NewNTUnicodeString(test.s)
		if (err != nil) != test.wantErr {
			t.Errorf("NewNTUnicodeString(%q): %v, wantErr:%v", test.s, err, test.wantErr)
			continue
		}
		if err != nil {
			continue
		}
		u16 := unsafe.Slice(ntus.Buffer, ntus.MaximumLength/2)[:ntus.Length/2]
		s2 := syscall.UTF16ToString(u16)
		if test.s != s2 {
			t.Errorf("round trip of %q = %q, wanted original", test.s, s2)
		}
	}
}
