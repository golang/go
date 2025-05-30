// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm)

package mime

import (
	"reflect"
	"testing"
)

func initMimeUnixTest(t *testing.T) {
	once.Do(initMime)
	err := loadMimeGlobsFile("testdata/test.types.globs2")
	if err != nil {
		t.Fatal(err)
	}

	loadMimeFile("testdata/test.types")
}

func TestTypeByExtensionUNIX(t *testing.T) {
	initMimeUnixTest(t)
	typeTests := map[string]string{
		".T1":       "application/test",
		".t2":       "text/test; charset=utf-8",
		".t3":       "document/test",
		".t4":       "example/test",
		".png":      "image/png",
		",v":        "",
		"~":         "",
		".foo?ar":   "",
		".foo*r":    "",
		".foo[1-3]": "",
		".foo1":     "example/glob-range",
		".foo2":     "example/glob-range",
		".foo3":     "example/glob-range",
	}

	for ext, want := range typeTests {
		val := TypeByExtension(ext)
		if val != want {
			t.Errorf("TypeByExtension(%q) = %q, want %q", ext, val, want)
		}
	}
}

func TestMimeExtension(t *testing.T) {
	initMimeUnixTest(t)

	tests := []struct {
		typ  string
		want []string
	}{
		{typ: "example/glob-range", want: []string{".foo1", ".foo2", ".foo3"}},
		{typ: "application/x-test-man", want: []string{".1", ".2", ".3"}},
	}

	for _, tt := range tests {
		got, err := ExtensionsByType(tt.typ)
		if err != nil {
			t.Errorf("ExtensionsByType(%q): %v", tt.typ, err)
			continue
		}
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("ExtensionsByType(%q) = %q; want %q", tt.typ, got, tt.want)
		}
	}
}

func Test_expansion(t *testing.T) {
	tests := []struct {
		glob string
		ok   bool
		want []string
	}{
		{
			glob: "foo",
			ok:   true,
			want: []string{"foo"},
		},
		{
			glob: ".foo[1-3da-c]",
			ok:   true,
			want: []string{".foo1", ".foo2", ".foo3", ".food", ".fooa", ".foob", ".fooc"},
		},
		{
			glob: ".foo[1-3][1-4]",
			ok:   false,
		},
		{
			glob: `.foo\[1-3`,
			ok:   true,
			want: []string{".foo[1-3"},
		},
		{
			glob: `.foo[1-3`,
			ok:   true,
			want: []string{".foo[1-3"},
		},
		{
			glob: ".foo1-3]",
			ok:   true,
			want: []string{".foo1-3]"},
		},
		{
			glob: ".foo[12-]",
			ok:   true,
			want: []string{".foo1", ".foo2", ".foo-"},
		},
		{
			glob: ".foo[-12]",
			ok:   true,
			want: []string{".foo-", ".foo1", ".foo2"},
		},
		{
			glob: ".foo[3-1]",
			ok:   false,
		},
		{
			glob: "foo[1-3].bar",
			ok:   true,
			want: []string{"foo1.bar", "foo2.bar", "foo3.bar"},
		},
		{
			glob: ".foo[!1-3]",
			ok:   false,
		},
		{
			glob: ".foo[!12]",
			ok:   false,
		},
		{
			glob: ".foo[1-3!a]",
			ok:   true,
			want: []string{".foo1", ".foo2", ".foo3", ".foo!", ".fooa"},
		},
		{
			glob: "[0-12-5]",
			ok:   true,
			want: []string{"0", "1", "2", "3", "4", "5"},
		},
		{
			glob: "[][!]",
			ok:   true,
			want: []string{"]", "[", "!"},
		},
		{
			glob: "[--0*?]",
			ok:   true,
			want: []string{"-", ".", "0", "*", "?"},
		},
		{
			glob: ".foo[]",
			ok:   true,
			want: []string{".foo[]"},
		},
		{
			glob: ".foo[1-3][4-5]",
			ok:   false,
		},
	}

	for _, tt := range tests {
		got, ok := expand(tt.glob)
		if ok != tt.ok {
			t.Errorf("expansion(%q) status = %v; want %v", tt.glob, ok, tt.ok)
		}

		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("expansion(%q) result = %q; want %q", tt.glob, got, tt.want)
		}
	}
}
