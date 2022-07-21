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
