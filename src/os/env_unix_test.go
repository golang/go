// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package os_test

import (
	"fmt"
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

var shellSpecialVarTests = []struct {
	k, v string
}{
	{"*", "asterisk"},
	{"#", "pound"},
	{"$", "dollar"},
	{"@", "at"},
	{"!", "exclamation mark"},
	{"?", "question mark"},
	{"-", "dash"},
}

func TestExpandEnvShellSpecialVar(t *testing.T) {
	for _, tt := range shellSpecialVarTests {
		Setenv(tt.k, tt.v)
		defer Unsetenv(tt.k)

		argRaw := fmt.Sprintf("$%s", tt.k)
		argWithBrace := fmt.Sprintf("${%s}", tt.k)
		if gotRaw, gotBrace := ExpandEnv(argRaw), ExpandEnv(argWithBrace); gotRaw != gotBrace {
			t.Errorf("ExpandEnv(%q) = %q, ExpandEnv(%q) = %q; expect them to be equal", argRaw, gotRaw, argWithBrace, gotBrace)
		}
	}
}
