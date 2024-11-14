// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	"bytes"
	"testing"
)

func FuzzEqualFold(f *testing.F) {
	for _, ss := range [][2]string{
		{"", ""},
		{"123abc", "123ABC"},
		{"αβδ", "ΑΒΔ"},
		{"abc", "xyz"},
		{"abc", "XYZ"},
		{"1", "2"},
		{"hello, world!", "hello, world!"},
		{"hello, world!", "Hello, World!"},
		{"hello, world!", "HELLO, WORLD!"},
		{"hello, world!", "jello, world!"},
		{"γειά, κόσμε!", "γειά, κόσμε!"},
		{"γειά, κόσμε!", "Γειά, Κόσμε!"},
		{"γειά, κόσμε!", "ΓΕΙΆ, ΚΌΣΜΕ!"},
		{"γειά, κόσμε!", "ΛΕΙΆ, ΚΌΣΜΕ!"},
		{"AESKey", "aesKey"},
		{"AESKEY", "aes_key"},
		{"aes_key", "AES_KEY"},
		{"AES_KEY", "aes-key"},
		{"aes-key", "AES-KEY"},
		{"AES-KEY", "aesKey"},
		{"aesKey", "AesKey"},
		{"AesKey", "AESKey"},
		{"AESKey", "aeskey"},
		{"DESKey", "aeskey"},
		{"AES Key", "aeskey"},
	} {
		f.Add([]byte(ss[0]), []byte(ss[1]))
	}
	equalFold := func(x, y []byte) bool { return string(foldName(x)) == string(foldName(y)) }
	f.Fuzz(func { t, x, y ->
		got := equalFold(x, y)
		want := bytes.EqualFold(x, y)
		if got != want {
			t.Errorf("equalFold(%q, %q) = %v, want %v", x, y, got, want)
		}
	})
}
