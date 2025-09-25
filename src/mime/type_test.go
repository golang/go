// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

import (
	"slices"
	"strings"
	"sync"
	"testing"
)

func setMimeInit(fn func()) (cleanup func()) {
	once = sync.Once{}
	testInitMime = fn
	return func() {
		testInitMime = nil
		once = sync.Once{}
	}
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
		if !slices.Equal(got, tt.want) {
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

func TestExtensionsByType2(t *testing.T) {
	cleanup := setMimeInit(func() {
		clearMimeTypes()
		// Initialize built-in types like in type.go before osInitMime.
		setMimeTypes(builtinTypesLower, builtinTypesLower)
	})
	defer cleanup()

	tests := []struct {
		typ  string
		want []string
	}{
		{typ: "application/postscript", want: []string{".ai", ".eps", ".ps"}},
		{typ: "application/vnd.android.package-archive", want: []string{".apk"}},
		{typ: "image/apng", want: []string{".apng"}},
		{typ: "image/avif", want: []string{".avif"}},
		{typ: "application/octet-stream", want: []string{".bin", ".com", ".exe"}},
		{typ: "image/bmp", want: []string{".bmp"}},
		{typ: "text/css; charset=utf-8", want: []string{".css"}},
		{typ: "text/csv; charset=utf-8", want: []string{".csv"}},
		{typ: "application/msword", want: []string{".doc"}},
		{typ: "application/vnd.openxmlformats-officedocument.wordprocessingml.document", want: []string{".docx"}},
		{typ: "text/html; charset=utf-8", want: []string{".ehtml", ".htm", ".html", ".shtml"}},
		{typ: "message/rfc822", want: []string{".eml"}},
		{typ: "audio/flac", want: []string{".flac"}},
		{typ: "image/gif", want: []string{".gif"}},
		{typ: "application/gzip", want: []string{".gz"}},
		{typ: "image/vnd.microsoft.icon", want: []string{".ico"}},
		{typ: "text/calendar; charset=utf-8", want: []string{".ics"}},
		{typ: "image/jpeg", want: []string{".jfif", ".jpeg", ".jpg", ".pjp", ".pjpeg"}},
		{typ: "text/javascript; charset=utf-8", want: []string{".js", ".mjs"}},
		{typ: "application/json", want: []string{".json"}},
		{typ: "audio/mp4", want: []string{".m4a"}},
		{typ: "audio/mpeg", want: []string{".mp3"}},
		{typ: "video/mp4", want: []string{".mp4"}},
		{typ: "audio/ogg", want: []string{".oga", ".ogg", ".opus"}},
		{typ: "video/ogg", want: []string{".ogv"}},
		{typ: "application/pdf", want: []string{".pdf"}},
		{typ: "image/png", want: []string{".png"}},
		{typ: "application/vnd.ms-powerpoint", want: []string{".ppt"}},
		{typ: "application/vnd.openxmlformats-officedocument.presentationml.presentation", want: []string{".pptx"}},
		{typ: "application/rdf+xml", want: []string{".rdf"}},
		{typ: "application/rtf", want: []string{".rtf"}},
		{typ: "image/svg+xml", want: []string{".svg"}},
		{typ: "text/plain; charset=utf-8", want: []string{".text", ".txt"}},
		{typ: "image/tiff", want: []string{".tif", ".tiff"}},
		{typ: "text/vtt; charset=utf-8", want: []string{".vtt"}},
		{typ: "application/wasm", want: []string{".wasm"}},
		{typ: "audio/wav", want: []string{".wav"}},
		{typ: "audio/webm", want: []string{".webm"}},
		{typ: "image/webp", want: []string{".webp"}},
		{typ: "text/xml; charset=utf-8", want: []string{".xbl", ".xml", ".xsl"}},
		{typ: "image/x-xbitmap", want: []string{".xbm"}},
		{typ: "application/xhtml+xml", want: []string{".xht", ".xhtml"}},
		{typ: "application/vnd.ms-excel", want: []string{".xls"}},
		{typ: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", want: []string{".xlsx"}},
		{typ: "application/zip", want: []string{".zip"}},
	}

	for _, tt := range tests {
		got, err := ExtensionsByType(tt.typ)
		if err != nil {
			t.Errorf("ExtensionsByType(%q): %v", tt.typ, err)
			continue
		}
		if !slices.Equal(got, tt.want) {
			t.Errorf("ExtensionsByType(%q) = %q; want %q", tt.typ, got, tt.want)
		}
	}
}
