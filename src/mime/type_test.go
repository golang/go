// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

import (
	"reflect"
	"strings"
	"sync"
	"testing"
)

func setMimeInit(fn func()) (cleanup func()) {
	once = sync.Once{}
	testInitMime = fn
	return func() { testInitMime = nil }
}

func clearMimeTypes() {
	setMimeTypes(map[string]string{}, map[string]string{})
}

func setType(ext, typ string) {
	if !strings.HasPrefix(ext, ".") {
		panic("missing leading dot")
	}
	if err := setExtensionType(ext, typ); err != nil {
		panic("bad test data: " + err.Error())
	}
}

func TestTypeByExtension(t *testing.T) {
	once = sync.Once{}
	// initMimeForTests returns the platform-specific extension =>
	// type tests. On Unix and Plan 9, this also tests the parsing
	// of MIME text files (in testdata/*). On Windows, we test the
	// real registry on the machine and assume that ".png" exists
	// there, which empirically it always has, for all versions of
	// Windows.
	typeTests := initMimeForTests()

	for ext, want := range typeTests {
		val := TypeByExtension(ext)
		if val != want {
			t.Errorf("TypeByExtension(%q) = %q, want %q", ext, val, want)
		}
	}
}

func TestTypeByExtension_LocalData(t *testing.T) {
	cleanup := setMimeInit(func() {
		clearMimeTypes()
		setType(".foo", "x/foo")
		setType(".bar", "x/bar")
		setType(".Bar", "x/bar; capital=1")
	})
	defer cleanup()

	tests := map[string]string{
		".foo":          "x/foo",
		".bar":          "x/bar",
		".Bar":          "x/bar; capital=1",
		".sdlkfjskdlfj": "",
		".t1":           "", // testdata shouldn't be used
	}

	for ext, want := range tests {
		val := TypeByExtension(ext)
		if val != want {
			t.Errorf("TypeByExtension(%q) = %q, want %q", ext, val, want)
		}
	}
}

func TestTypeByExtensionCase(t *testing.T) {
	const custom = "test/test; charset=iso-8859-1"
	const caps = "test/test; WAS=ALLCAPS"

	cleanup := setMimeInit(func() {
		clearMimeTypes()
		setType(".TEST", caps)
		setType(".tesT", custom)
	})
	defer cleanup()

	// case-sensitive lookup
	if got := TypeByExtension(".tesT"); got != custom {
		t.Fatalf("for .tesT, got %q; want %q", got, custom)
	}
	if got := TypeByExtension(".TEST"); got != caps {
		t.Fatalf("for .TEST, got %q; want %s", got, caps)
	}

	// case-insensitive
	if got := TypeByExtension(".TesT"); got != custom {
		t.Fatalf("for .TesT, got %q; want %q", got, custom)
	}
}

func TestExtensionsByType(t *testing.T) {
	cleanup := setMimeInit(func() {
		clearMimeTypes()
		setType(".gif", "image/gif")
		setType(".a", "foo/letter")
		setType(".b", "foo/letter")
		setType(".B", "foo/letter")
		setType(".PNG", "image/png")
	})
	defer cleanup()

	tests := []struct {
		typ     string
		want    []string
		wantErr string
	}{
		{typ: "image/gif", want: []string{".gif"}},
		{typ: "image/png", want: []string{".png"}}, // lowercase
		{typ: "foo/letter", want: []string{".a", ".b"}},
		{typ: "x/unknown", want: nil},
	}

	for _, tt := range tests {
		got, err := ExtensionsByType(tt.typ)
		if err != nil && tt.wantErr != "" && strings.Contains(err.Error(), tt.wantErr) {
			continue
		}
		if err != nil {
			t.Errorf("ExtensionsByType(%q) error: %v", tt.typ, err)
			continue
		}
		if tt.wantErr != "" {
			t.Errorf("ExtensionsByType(%q) = %q, %v; want error substring %q", tt.typ, got, err, tt.wantErr)
			continue
		}
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("ExtensionsByType(%q) = %q; want %q", tt.typ, got, tt.want)
		}
	}
}

func TestLookupMallocs(t *testing.T) {
	n := testing.AllocsPerRun(10000, func() {
		TypeByExtension(".html")
		TypeByExtension(".HtML")
	})
	if n > 0 {
		t.Errorf("allocs = %v; want 0", n)
	}
}

func BenchmarkTypeByExtension(b *testing.B) {
	initMime()
	b.ResetTimer()

	for _, ext := range []string{
		".html",
		".HTML",
		".unused",
	} {
		b.Run(ext, func(b *testing.B) {
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					TypeByExtension(ext)
				}
			})
		})
	}
}

func BenchmarkExtensionsByType(b *testing.B) {
	initMime()
	b.ResetTimer()

	for _, typ := range []string{
		"text/html",
		"text/html; charset=utf-8",
		"application/octet-stream",
	} {
		b.Run(typ, func(b *testing.B) {
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					if _, err := ExtensionsByType(typ); err != nil {
						b.Fatal(err)
					}
				}
			})
		})
	}
}
