// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd

package os_test

import (
	. "os"
	"testing"
)

var setenvEinvalTests = []struct {
	k, v string
}{
	{"", ""},      // empty key
	{"k=v", ""},   // '=' in key
	{"\x00", ""},  // '\x00' in key
	{"k", "\x00"}, // '\x00' in value
}

func TestSetenvUnixEinval(t *testing.T) {
	for _, tt := range setenvEinvalTests {
		err := Setenv(tt.k, tt.v)
		if err == nil {
			t.Errorf(`Setenv(%q, %q) == nil, want error`, tt.k, tt.v)
		}
	}
}
