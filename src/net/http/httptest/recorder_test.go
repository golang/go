// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httptest

import (
	"fmt"
	"io"
	"net/http"
	"testing"
)

func TestRecorder(t *testing.T) {
	type checkFunc func(*ResponseRecorder) error
	check := func(fns ...checkFunc) []checkFunc { return fns }

	hasStatus := func(wantCode int) checkFunc {
		return func(rec *ResponseRecorder) error {
			if rec.Code != wantCode {
				return fmt.Errorf("Status = %d; want %d", rec.Code, wantCode)
			}
			return nil
		}
	}
	hasResultStatus := func(want string) checkFunc {
		return func(rec *ResponseRecorder) error {
			if rec.Result().Status != want {
				return fmt.Errorf("Result().Status = %q; want %q", rec.Result().Status, want)
			}
			return nil
		}
	}
	hasResultStatusCode := func(wantCode int) checkFunc {
		return func(rec *ResponseRecorder) error {
			if rec.Result().StatusCode != wantCode {
				return fmt.Errorf("Result().StatusCode = %d; want %d", rec.Result().StatusCode, wantCode)
			}
			return nil
		}
	}
	hasResultContents := func(want string) checkFunc {
		return func(rec *ResponseRecorder) error {
			contentBytes, err := io.ReadAll(rec.Result().Body)
			if err != nil {
				return err
			}
			contents := string(contentBytes)
			if contents != want {
				return fmt.Errorf("Result().Body = %s; want %s", contents, want)
			}
			return nil
		}
	}
	hasContents := func(want string) checkFunc {
		return func(rec *ResponseRecorder) error {
			if rec.Body.String() != want {
				return fmt.Errorf("wrote = %q; want %q", rec.Body.String(), want)
			}
			return nil
		}
	}
	hasFlush := func(want bool) checkFunc {
		return func(rec *ResponseRecorder) error {
			if rec.Flushed != want {
				return fmt.Errorf("Flushed = %v; want %v", rec.Flushed, want)
			}
			return nil
		}
	}
	hasOldHeader := func(key, want string) checkFunc {
		return func(rec *ResponseRecorder) error {
			if got := rec.HeaderMap.Get(key); got != want {
				return fmt.Errorf("HeaderMap header %s = %q; want %q", key, got, want)
			}
			return nil
		}
	}
	hasHeader := func(key, want string) checkFunc {
		return func(rec *ResponseRecorder) error {
			if got := rec.Result().Header.Get(key); got != want {
				return fmt.Errorf("final header %s = %q; want %q", key, got, want)
			}
			return nil
		}
	}
	hasNotHeaders := func(keys ...string) checkFunc {
		return func(rec *ResponseRecorder) error {
			for _, k := range keys {
				v, ok := rec.Result().Header[http.CanonicalHeaderKey(k)]
				if ok {
					return fmt.Errorf("unexpected header %s with value %q", k, v)
				}
			}
			return nil
		}
	}
	hasTrailer := func(key, want string) checkFunc {
		return func(rec *ResponseRecorder) error {
			if got := rec.Result().Trailer.Get(key); got != want {
				return fmt.Errorf("trailer %s = %q; want %q", key, got, want)
			}
			return nil
		}
	}
	hasNotTrailers := func(keys ...string) checkFunc {
		return func(rec *ResponseRecorder) error {
			trailers := rec.Result().Trailer
			for _, k := range keys {
				_, ok := trailers[http.CanonicalHeaderKey(k)]
				if ok {
					return fmt.Errorf("unexpected trailer %s", k)
				}
			}
			return nil
		}
	}
	hasContentLength := func(length int64) checkFunc {
		return func(rec *ResponseRecorder) error {
			if got := rec.Result().ContentLength; got != length {
				return fmt.Errorf("ContentLength = %d; want %d", got, length)
			}
			return nil
		}
	}

	for _, tt := range [...]struct {
		name   string
		h      func(w http.ResponseWriter, r *http.Request)
		checks []checkFunc
	}{
		{
			"200 default",
			func(w http.ResponseWriter, r *http.Request) {},
			check(hasStatus(200), hasContents("")),
		},
		{
			"first code only",
			func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(201)
				w.WriteHeader(202)
				w.Write([]byte("hi"))
			},
			check(hasStatus(201), hasContents("hi")),
		},
		{
			"write sends 200",
			func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte("hi first"))
				w.WriteHeader(201)
				w.WriteHeader(202)
			},
			check(hasStatus(200), hasContents("hi first"), hasFlush(false)),
		},
		{
			"write string",
			func(w http.ResponseWriter, r *http.Request) {
				io.WriteString(w, "hi first")
			},
			check(
				hasStatus(200),
				hasContents("hi first"),
				hasFlush(false),
				hasHeader("Content-Type", "text/plain; charset=utf-8"),
			),
		},
		{
			"flush",
			func(w http.ResponseWriter, r *http.Request) {
				w.(http.Flusher).Flush() // also sends a 200
				w.WriteHeader(201)
			},
			check(hasStatus(200), hasFlush(true), hasContentLength(-1)),
		},
		{
			"Content-Type detection",
			func(w http.ResponseWriter, r *http.Request) {
				io.WriteString(w, "<html>")
			},
			check(hasHeader("Content-Type", "text/html; charset=utf-8")),
		},
		{
			"no Content-Type detection with Transfer-Encoding",
			func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Transfer-Encoding", "some encoding")
				io.WriteString(w, "<html>")
			},
			check(hasHeader("Content-Type", "")), // no header
		},
		{
			"no Content-Type detection if set explicitly",
			func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "some/type")
				io.WriteString(w, "<html>")
			},
			check(hasHeader("Content-Type", "some/type")),
		},
		{
			"Content-Type detection doesn't crash if HeaderMap is nil",
			func(w http.ResponseWriter, r *http.Request) {
				// Act as if the user wrote new(httptest.ResponseRecorder)
				// rather than using NewRecorder (which initializes
				// HeaderMap)
				w.(*ResponseRecorder).HeaderMap = nil
				io.WriteString(w, "<html>")
			},
			check(hasHeader("Content-Type", "text/html; charset=utf-8")),
		},
		{
			"Header is not changed after write",
			func(w http.ResponseWriter, r *http.Request) {
				hdr := w.Header()
				hdr.Set("Key", "correct")
				w.WriteHeader(200)
				hdr.Set("Key", "incorrect")
			},
			check(hasHeader("Key", "correct")),
		},
		{
			"Trailer headers are correctly recorded",
			func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Non-Trailer", "correct")
				w.Header().Set("Trailer", "Trailer-A")
				w.Header().Add("Trailer", "Trailer-B")
				w.Header().Add("Trailer", "Trailer-C")
				io.WriteString(w, "<html>")
				w.Header().Set("Non-Trailer", "incorrect")
				w.Header().Set("Trailer-A", "valuea")
				w.Header().Set("Trailer-C", "valuec")
				w.Header().Set("Trailer-NotDeclared", "should be omitted")
				w.Header().Set("Trailer:Trailer-D", "with prefix")
			},
			check(
				hasStatus(200),
				hasHeader("Content-Type", "text/html; charset=utf-8"),
				hasHeader("Non-Trailer", "correct"),
				hasNotHeaders("Trailer-A", "Trailer-B", "Trailer-C", "Trailer-NotDeclared"),
				hasTrailer("Trailer-A", "valuea"),
				hasTrailer("Trailer-C", "valuec"),
				hasNotTrailers("Non-Trailer", "Trailer-B", "Trailer-NotDeclared"),
				hasTrailer("Trailer-D", "with prefix"),
			),
		},
		{
			"Header set without any write", // Issue 15560
			func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("X-Foo", "1")

				// Simulate somebody using
				// new(ResponseRecorder) instead of
				// using the constructor which sets
				// this to 200
				w.(*ResponseRecorder).Code = 0
			},
			check(
				hasOldHeader("X-Foo", "1"),
				hasStatus(0),
				hasHeader("X-Foo", "1"),
				hasResultStatus("200 OK"),
				hasResultStatusCode(200),
			),
		},
		{
			"HeaderMap vs FinalHeaders", // more for Issue 15560
			func(w http.ResponseWriter, r *http.Request) {
				h := w.Header()
				h.Set("X-Foo", "1")
				w.Write([]byte("hi"))
				h.Set("X-Foo", "2")
				h.Set("X-Bar", "2")
			},
			check(
				hasOldHeader("X-Foo", "2"),
				hasOldHeader("X-Bar", "2"),
				hasHeader("X-Foo", "1"),
				hasNotHeaders("X-Bar"),
			),
		},
		{
			"setting Content-Length header",
			func(w http.ResponseWriter, r *http.Request) {
				body := "Some body"
				contentLength := fmt.Sprintf("%d", len(body))
				w.Header().Set("Content-Length", contentLength)
				io.WriteString(w, body)
			},
			check(hasStatus(200), hasContents("Some body"), hasContentLength(9)),
		},
		{
			"nil ResponseRecorder.Body", // Issue 26642
			func(w http.ResponseWriter, r *http.Request) {
				w.(*ResponseRecorder).Body = nil
				io.WriteString(w, "hi")
			},
			check(hasResultContents("")), // check we don't crash reading the body

		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, _ := http.NewRequest("GET", "http://foo.com/", nil)
			h := http.HandlerFunc(tt.h)
			rec := NewRecorder()
			h.ServeHTTP(rec, r)
			for _, check := range tt.checks {
				if err := check(rec); err != nil {
					t.Error(err)
				}
			}
		})
	}
}

// issue 39017 - disallow Content-Length values such as "+3"
func TestParseContentLength(t *testing.T) {
	tests := []struct {
		cl   string
		want int64
	}{
		{
			cl:   "3",
			want: 3,
		},
		{
			cl:   "+3",
			want: -1,
		},
		{
			cl:   "-3",
			want: -1,
		},
		{
			// max int64, for safe conversion before returning
			cl:   "9223372036854775807",
			want: 9223372036854775807,
		},
		{
			cl:   "9223372036854775808",
			want: -1,
		},
	}

	for _, tt := range tests {
		if got := parseContentLength(tt.cl); got != tt.want {
			t.Errorf("%q:\n\tgot=%d\n\twant=%d", tt.cl, got, tt.want)
		}
	}
}
