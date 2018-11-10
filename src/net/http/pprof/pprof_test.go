// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pprof

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestHandlers(t *testing.T) {
	testCases := []struct {
		path               string
		handler            http.HandlerFunc
		statusCode         int
		contentType        string
		contentDisposition string
		resp               []byte
	}{
		{"/debug/pprof/<script>scripty<script>", Index, http.StatusNotFound, "text/plain; charset=utf-8", "", []byte("Unknown profile\n")},
		{"/debug/pprof/heap", Index, http.StatusOK, "application/octet-stream", `attachment; filename="heap"`, nil},
		{"/debug/pprof/heap?debug=1", Index, http.StatusOK, "text/plain; charset=utf-8", "", nil},
		{"/debug/pprof/cmdline", Cmdline, http.StatusOK, "text/plain; charset=utf-8", "", nil},
		{"/debug/pprof/profile?seconds=1", Profile, http.StatusOK, "application/octet-stream", `attachment; filename="profile"`, nil},
		{"/debug/pprof/symbol", Symbol, http.StatusOK, "text/plain; charset=utf-8", "", nil},
		{"/debug/pprof/trace", Trace, http.StatusOK, "application/octet-stream", `attachment; filename="trace"`, nil},
	}
	for _, tc := range testCases {
		t.Run(tc.path, func(t *testing.T) {
			req := httptest.NewRequest("GET", "http://example.com"+tc.path, nil)
			w := httptest.NewRecorder()
			tc.handler(w, req)

			resp := w.Result()
			if got, want := resp.StatusCode, tc.statusCode; got != want {
				t.Errorf("status code: got %d; want %d", got, want)
			}

			body, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				t.Errorf("when reading response body, expected non-nil err; got %v", err)
			}
			if got, want := resp.Header.Get("X-Content-Type-Options"), "nosniff"; got != want {
				t.Errorf("X-Content-Type-Options: got %q; want %q", got, want)
			}
			if got, want := resp.Header.Get("Content-Type"), tc.contentType; got != want {
				t.Errorf("Content-Type: got %q; want %q", got, want)
			}
			if got, want := resp.Header.Get("Content-Disposition"), tc.contentDisposition; got != want {
				t.Errorf("Content-Disposition: got %q; want %q", got, want)
			}

			if resp.StatusCode == http.StatusOK {
				return
			}
			if got, want := resp.Header.Get("X-Go-Pprof"), "1"; got != want {
				t.Errorf("X-Go-Pprof: got %q; want %q", got, want)
			}
			if !bytes.Equal(body, tc.resp) {
				t.Errorf("response: got %q; want %q", body, tc.resp)
			}
		})
	}

}
