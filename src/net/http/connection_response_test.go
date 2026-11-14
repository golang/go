// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http_test

import (
	"bufio"
	"bytes"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
)

func TestReadResponsePreservesConnection(t *testing.T) {
	for _, tt := range []struct {
		name, protocol string
		values         []string
		close          bool
	}{
		{"closeLast", "HTTP/1.1", []string{"X-Hop, close"}, true},
		{"closeFirst", "HTTP/1.1", []string{"close, X-Hop"}, true},
		{"multipleLines", "HTTP/1.1", []string{"X-Hop", "close, X-Other"}, true},
		{"mixedCase", "HTTP/1.1", []string{"x-hOP, ClOsE"}, true},
		{"keepAlive", "HTTP/1.1", []string{"X-Hop, keep-alive"}, false},
		{"HTTP10", "HTTP/1.0", []string{"X-Hop, close"}, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			wire := tt.protocol + " 200 OK\r\n"
			for _, v := range tt.values {
				wire += "Connection: " + v + "\r\n"
			}
			wire += "X-Hop: secret\r\nX-Extension: keep\r\nContent-Length: 2\r\n\r\nok"
			res, err := http.ReadResponse(bufio.NewReader(strings.NewReader(wire)), nil)
			if err != nil {
				t.Fatal(err)
			}
			defer res.Body.Close()
			if !reflect.DeepEqual(res.Header.Values("Connection"), tt.values) {
				t.Fatalf("Connection=%q want=%q", res.Header.Values("Connection"), tt.values)
			}
			if res.Close != tt.close || res.Header.Get("X-Hop") != "secret" || res.Header.Get("X-Extension") != "keep" {
				t.Fatalf("Close=%v Header=%v", res.Close, res.Header)
			}
			b, err := io.ReadAll(res.Body)
			if err != nil || string(b) != "ok" {
				t.Fatalf("body=%q error=%v", b, err)
			}
		})
	}
}

func TestResponseWriteMultipleConnectionClose(t *testing.T) {
	for _, tt := range []struct {
		name               string
		values, want       []string
		close, parsedClose bool
	}{
		{"secondValue", []string{"X-Hop", "close"}, []string{"X-Hop", "close"}, true, true},
		{"firstValue", []string{"close", "X-Hop"}, []string{"close", "X-Hop"}, true, true},
		{"mixedCase", []string{"X-Hop", "ClOsE, X-Other"}, []string{"X-Hop", "ClOsE, X-Other"}, true, true},
		{"combined", []string{"X-Hop, close"}, []string{"X-Hop, close"}, true, true},
		{"syntheticClose", nil, []string{"close"}, true, true},
		{"keepAlive", []string{"X-Hop, keep-alive"}, []string{"X-Hop, keep-alive"}, false, false},
		{"noConnection", nil, nil, false, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			res := &http.Response{StatusCode: 200, ProtoMajor: 1, ProtoMinor: 1, Header: http.Header{"X-Hop": {"secret"}}, Close: tt.close, ContentLength: 2, Body: io.NopCloser(strings.NewReader("ok"))}
			if tt.values != nil {
				res.Header["Connection"] = append([]string(nil), tt.values...)
			}
			var buf bytes.Buffer
			if err := res.Write(&buf); err != nil {
				t.Fatal(err)
			}
			parsed, err := http.ReadResponse(bufio.NewReader(&buf), nil)
			if err != nil {
				t.Fatal(err)
			}
			defer parsed.Body.Close()
			if !reflect.DeepEqual(parsed.Header.Values("Connection"), tt.want) || parsed.Close != tt.parsedClose || res.Close != tt.close {
				t.Fatalf("Header=%v Close=%v original Close=%v", parsed.Header, parsed.Close, res.Close)
			}
			body, err := io.ReadAll(parsed.Body)
			if err != nil || string(body) != "ok" || parsed.Header.Get("X-Hop") != "secret" {
				t.Fatalf("body=%q err=%v header=%v", body, err, parsed.Header)
			}
		})
	}
}
