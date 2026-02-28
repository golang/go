// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm)

package mime

import (
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
	}

	for ext, want := range typeTests {
		val := TypeByExtension(ext)
		if val != want {
			t.Errorf("TypeByExtension(%q) = %q, want %q", ext, val, want)
		}
	}
}
