// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import "testing"

func TestLogosCloneBaseName(t *testing.T) {
	checks := []struct {
		label string
		input string
		want  string
	}{
		{label: "comptimeOne", input: "main.Autowire·logosct7", want: "main.Autowire"},
		{label: "comptimeLongNumber", input: "main.Autowire·logosct4096", want: "main.Autowire"},
		{label: "comptimeZero", input: "main.Autowire·logosct0", want: "main.Autowire"},
		{label: "comptimePackagePath", input: "example.com/p.Autowire·logosct123", want: "example.com/p.Autowire"},
		{label: "pointerReceiver", input: "example.com/p.(*T).Method·logosct2", want: "example.com/p.(*T).Method"},
		{label: "valueReceiver", input: "example.com/p.T.Method·logosct12", want: "example.com/p.T.Method"},
		{label: "genericComptime", input: "example.com/p.Foo[go.shape.string]·logosct10", want: "example.com/p.Foo[go.shape.string]"},
		{label: "comptimeASCIISuffix", input: "example.com/p.Foo_logosct1", want: "example.com/p.Foo"},
		{label: "comptimeASCIIPackagePath", input: "example.com/p.Autowire_logosct123", want: "example.com/p.Autowire"},
		{label: "comptimeASCIIReceiver", input: "example.com/p.T.Method_logosct12", want: "example.com/p.T.Method"},
		{label: "comptimeASCIIGeneric", input: "example.com/p.Foo[go.shape.string]_logosct10", want: "example.com/p.Foo[go.shape.string]"},
		{label: "comptimeASCIINumberMissing", input: "example.com/p.Foo_logosct", want: "example.com/p.Foo_logosct"},
		{label: "comptimeASCIINoBase", input: "_logosct1", want: "_logosct1"},
		{label: "comptimeASCIILetterNumber", input: "example.com/p.Foo_logosctX", want: "example.com/p.Foo_logosctX"},
		{label: "comptimeASCIILetterAfterNumber", input: "example.com/p.Foo_logosct1x", want: "example.com/p.Foo_logosct1x"},
		{label: "comptimeASCIIDigitsAfterOtherText", input: "example.com/p.Foo_logosct2extra3", want: "example.com/p.Foo_logosct2extra3"},
		{label: "ordinaryFunction", input: "main.main", want: "main.main"},
		{label: "abiWrapper", input: "runtime.cgocall.abi0", want: "runtime.cgocall.abi0"},
		{label: "ordinaryMethodName", input: "main.Autowire", want: "main.Autowire"},
		{label: "asciiPeriod", input: "example.com/p.Foo.logosct1", want: "example.com/p.Foo.logosct1"},
		{label: "numberMissing", input: "example.com/p.Foo·logosct", want: "example.com/p.Foo·logosct"},
		{label: "letterNumber", input: "example.com/p.Foo·logosctX", want: "example.com/p.Foo·logosctX"},
		{label: "hexLikeNumber", input: "example.com/p.Foo·logosct0x1", want: "example.com/p.Foo·logosct0x1"},
		{label: "signedNumber", input: "example.com/p.Foo·logosct-1", want: "example.com/p.Foo·logosct-1"},
		{label: "noBaseComptime", input: "·logosct1", want: "·logosct1"},
		{label: "letterAfterNumber", input: "example.com/p.Foo·logosct1x", want: "example.com/p.Foo·logosct1x"},
		{label: "underscoreAfterNumber", input: "example.com/p.Foo·logosct1_extra", want: "example.com/p.Foo·logosct1_extra"},
		{label: "middleDotTextAfterNumber", input: "example.com/p.Foo·logosct1·extra", want: "example.com/p.Foo·logosct1·extra"},
		{label: "digitsAfterOtherText", input: "example.com/p.Foo·logosct2extra3", want: "example.com/p.Foo·logosct2extra3"},
		{label: "dotAfterNumber", input: "example.com/p.Foo·logosct1.2", want: "example.com/p.Foo·logosct1.2"},
	}

	for _, tc := range checks {
		t.Run(tc.label, func(t *testing.T) {
			if got := logosCloneBaseName(tc.input); got != tc.want {
				t.Fatalf("logosCloneBaseName(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}
