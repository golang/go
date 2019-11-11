// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests of internal functions and things with no better homes.

package http

import (
	"bytes"
	"internal/testenv"
	"net/url"
	"os/exec"
	"reflect"
	"testing"
	"time"
)

func init() {
	shutdownPollInterval = 5 * time.Millisecond
}

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
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("foreachHeaderElement(%q) = %q; want %q", tt.in, got, tt.want)
		}
	}
}

func TestCleanHost(t *testing.T) {
	tests := []struct {
		in, want string
	}{
		{"www.google.com", "www.google.com"},
		{"www.google.com foo", "www.google.com"},
		{"www.google.com/foo", "www.google.com"},
		{" first character is a space", ""},
		{"[1::6]:8080", "[1::6]:8080"},

		// Punycode:
		{"гофер.рф/foo", "xn--c1ae0ajs.xn--p1ai"},
		{"bücher.de", "xn--bcher-kva.de"},
		{"bücher.de:8080", "xn--bcher-kva.de:8080"},
		// Verify we convert to lowercase before punycode:
		{"BÜCHER.de", "xn--bcher-kva.de"},
		{"BÜCHER.de:8080", "xn--bcher-kva.de:8080"},
		// Verify we normalize to NFC before punycode:
		{"gophér.nfc", "xn--gophr-esa.nfc"},            // NFC input; no work needed
		{"goph\u0065\u0301r.nfd", "xn--gophr-esa.nfd"}, // NFD input
	}
	for _, tt := range tests {
		got := cleanHost(tt.in)
		if tt.want != got {
			t.Errorf("cleanHost(%q) = %q, want %q", tt.in, got, tt.want)
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
	out, err := exec.Command(goBin, "tool", "nm", goBin).CombinedOutput()
	if err != nil {
		t.Fatalf("go tool nm: %v: %s", err, out)
	}
	wantSym := map[string]bool{
		// Verify these exist: (sanity checking this test)
		"net/http.(*Client).Get":          true,
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
	out, err := exec.Command(goTool, "test", "-short", "-tags=nethttpomithttp2", "net/http").CombinedOutput()
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
	out, err := exec.Command(goTool, "vet", "-tags=nethttpomithttp2", "net/http").CombinedOutput()
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
