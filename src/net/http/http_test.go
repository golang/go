// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests of internal functions and things with no better homes.

package http

import (
	"bytes"
	"internal/testenv"
	"io"
	"io/fs"
	"net/url"
	"os"
	"regexp"
	"slices"
	"strings"
	"testing"
)

func TestForeachHeaderElement(t *testing.T) {
	tests := []struct {
		in   string
		want []string
	}{
		{"Foo", []string{"Foo"}},
		{" Foo", []string{"Foo"}},
		{"Foo ", []string{"Foo"}},
		{" Foo ", []string{"Foo"}},

		{"foo", []string{"foo"}},
		{"anY-cAsE", []string{"anY-cAsE"}},

		{"", nil},
		{",,,,  ,  ,,   ,,, ,", nil},

		{" Foo,Bar, Baz,lower,,Quux ", []string{"Foo", "Bar", "Baz", "lower", "Quux"}},
	}
	for _, tt := range tests {
		var got []string
		foreachHeaderElement(tt.in, func(v string) {
			got = append(got, v)
		})
		if !slices.Equal(got, tt.want) {
			t.Errorf("foreachHeaderElement(%q) = %q; want %q", tt.in, got, tt.want)
		}
	}
}

// Test that cmd/go doesn't link in the HTTP server.
//
// This catches accidental dependencies between the HTTP transport and
// server code.
func TestCmdGoNoHTTPServer(t *testing.T) {
	t.Parallel()
	goBin := testenv.GoToolPath(t)
	out, err := testenv.Command(t, goBin, "tool", "nm", goBin).CombinedOutput()
	if err != nil {
		t.Fatalf("go tool nm: %v: %s", err, out)
	}
	wantSym := map[string]bool{
		// Verify these exist: (sanity checking this test)
		"net/http.(*Client).do":           true,
		"net/http.(*Transport).RoundTrip": true,

		// Verify these don't exist:
		"net/http.http2Server":           false,
		"net/http.(*Server).Serve":       false,
		"net/http.(*ServeMux).ServeHTTP": false,
		"net/http.DefaultServeMux":       false,
	}
	for sym, want := range wantSym {
		got := bytes.Contains(out, []byte(sym))
		if !want && got {
			t.Errorf("cmd/go unexpectedly links in HTTP server code; found symbol %q in cmd/go", sym)
		}
		if want && !got {
			t.Errorf("expected to find symbol %q in cmd/go; not found", sym)
		}
	}
}

// Tests that the nethttpomithttp2 build tag doesn't rot too much,
// even if there's not a regular builder on it.
func TestOmitHTTP2(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	t.Parallel()
	goTool := testenv.GoToolPath(t)
	out, err := testenv.Command(t, goTool, "test", "-short", "-tags=nethttpomithttp2", "net/http").CombinedOutput()
	if err != nil {
		t.Fatalf("go test -short failed: %v, %s", err, out)
	}
}

// Tests that the nethttpomithttp2 build tag at least type checks
// in short mode.
// The TestOmitHTTP2 test above actually runs tests (in long mode).
func TestOmitHTTP2Vet(t *testing.T) {
	t.Parallel()
	goTool := testenv.GoToolPath(t)
	out, err := testenv.Command(t, goTool, "vet", "-tags=nethttpomithttp2", "net/http").CombinedOutput()
	if err != nil {
		t.Fatalf("go vet failed: %v, %s", err, out)
	}
}

var valuesCount int

func BenchmarkCopyValues(b *testing.B) {
	b.ReportAllocs()
	src := url.Values{
		"a": {"1", "2", "3", "4", "5"},
		"b": {"2", "2", "3", "4", "5"},
		"c": {"3", "2", "3", "4", "5"},
		"d": {"4", "2", "3", "4", "5"},
		"e": {"1", "1", "2", "3", "4", "5", "6", "7", "abcdef", "l", "a", "b", "c", "d", "z"},
		"j": {"1", "2"},
		"m": nil,
	}
	for i := 0; i < b.N; i++ {
		dst := url.Values{"a": {"b"}, "b": {"2"}, "c": {"3"}, "d": {"4"}, "j": nil, "m": {"x"}}
		copyValues(dst, src)
		if valuesCount = len(dst["a"]); valuesCount != 6 {
			b.Fatalf(`%d items in dst["a"] but expected 6`, valuesCount)
		}
	}
	if valuesCount == 0 {
		b.Fatal("Benchmark wasn't run")
	}
}

var forbiddenStringsFunctions = map[string]bool{
	// Functions that use Unicode-aware case folding.
	"EqualFold":      true,
	"Title":          true,
	"ToLower":        true,
	"ToLowerSpecial": true,
	"ToTitle":        true,
	"ToTitleSpecial": true,
	"ToUpper":        true,
	"ToUpperSpecial": true,

	// Functions that use Unicode-aware spaces.
	"Fields":    true,
	"TrimSpace": true,
}

// TestNoUnicodeStrings checks that nothing in net/http uses the Unicode-aware
// strings and bytes package functions. HTTP is mostly ASCII based, and doing
// Unicode-aware case folding or space stripping can introduce vulnerabilities.
func TestNoUnicodeStrings(t *testing.T) {
	testenv.MustHaveSource(t)

	re := regexp.MustCompile(`(strings|bytes).([A-Za-z]+)`)
	if err := fs.WalkDir(os.DirFS("."), ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			t.Fatal(err)
		}

		if path == "internal/ascii" {
			return fs.SkipDir
		}
		if !strings.HasSuffix(path, ".go") ||
			strings.HasSuffix(path, "_test.go") ||
			path == "h2_bundle.go" ||
			path == "internal/httpcommon/httpcommon.go" ||
			d.IsDir() {
			return nil
		}

		contents, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		for lineNum, line := range strings.Split(string(contents), "\n") {
			for _, match := range re.FindAllStringSubmatch(line, -1) {
				if !forbiddenStringsFunctions[match[2]] {
					continue
				}
				t.Errorf("disallowed call to %s at %s:%d", match[0], path, lineNum+1)
			}
		}

		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

type requestTooLargerResponseWriter struct {
	called bool
}

func (rw *requestTooLargerResponseWriter) Header() Header              { return Header{} }
func (rw *requestTooLargerResponseWriter) Write(b []byte) (int, error) { return len(b), nil }
func (rw *requestTooLargerResponseWriter) WriteHeader(statusCode int)  {}
func (rw *requestTooLargerResponseWriter) requestTooLarge() {
	rw.called = true
}

type wrapper struct {
	ResponseWriter
}

func (w *wrapper) Unwrap() ResponseWriter {
	return w.ResponseWriter
}

func TestMaxBytesReaderUnwrapTriggersRequestTooLarge(t *testing.T) {
	body := strings.NewReader("123456")
	limit := int64(5)

	innerRw := &requestTooLargerResponseWriter{}
	wrappedRw := &wrapper{ResponseWriter: innerRw}

	l := MaxBytesReader(wrappedRw, io.NopCloser(body), limit)

	buf := make([]byte, 10)
	_, err := l.Read(buf)

	if _, ok := err.(*MaxBytesError); !ok {
		t.Errorf("expected MaxBytesError, got %T", err)
	}
	if !innerRw.called {
		t.Errorf("expected requestTooLarge to be called, but it wasn't")
	}
}

func TestProtocols(t *testing.T) {
	var p Protocols
	if p.HTTP1() {
		t.Errorf("zero-value protocols: p.HTTP1() = true, want false")
	}
	p.SetHTTP1(true)
	p.SetHTTP2(true)
	if !p.HTTP1() {
		t.Errorf("initialized protocols: p.HTTP1() = false, want true")
	}
	if !p.HTTP2() {
		t.Errorf("initialized protocols: p.HTTP2() = false, want true")
	}
	p.SetHTTP1(false)
	if p.HTTP1() {
		t.Errorf("after unsetting HTTP1: p.HTTP1() = true, want false")
	}
	if !p.HTTP2() {
		t.Errorf("after unsetting HTTP1: p.HTTP2() = false, want true")
	}
}

const redirectURL = "/thisaredirect细雪withasciilettersのけぶabcdefghijk.html"

func BenchmarkHexEscapeNonASCII(b *testing.B) {
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		hexEscapeNonASCII(redirectURL)
	}
}
