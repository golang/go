// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2_test

import (
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
)

// TestFrameReuseEndToEndStress drives concurrent multiplexed requests
// through the real net/http HTTP/2 server and Transport (both of
// which call SetReuseFrames on the per-connection Framer) and touches
// every field reachable from a request/response in ways that would
// race against the read loop's next ReadFrame if anything still
// aliased the cached frame after the readMore gate (server) or
// read-loop iteration (Transport).
//
// Under -race this is a regression test against future refactors of
// processHeaders / handleResponse / processTrailers / processData
// that fail to copy aliased data. The synthetic Framer-pair tests in
// frame_reuse_race_test.go cannot reach the production process* code
// paths; this one does.
//
// The handler and client iterate Name/Value byte-by-byte so that any
// aliasing leak shows up as a concrete read concurrent with a write
// in bytes.(*Reader).Read inside ReadFrame.
func TestFrameReuseEndToEndStress(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping stress test in short mode")
	}

	const responseBody = "the quick brown fox jumps over the lazy dog"

	touchHeader := func(h http.Header) byte {
		var sink byte
		for k, vs := range h {
			for i := 0; i < len(k); i++ {
				sink ^= k[i]
			}
			for _, v := range vs {
				for i := 0; i < len(v); i++ {
					sink ^= v[i]
				}
			}
		}
		return sink
	}

	ts := newTestServer(t,
		func(w http.ResponseWriter, r *http.Request) {
			_ = touchHeader(r.Header)
			if _, err := io.Copy(io.Discard, r.Body); err != nil {
				t.Errorf("server body copy: %v", err)
				return
			}
			_ = touchHeader(r.Trailer)
			w.Header().Set("Trailer", "X-Server-Trailer")
			w.Header().Set("X-Response-Header", "response-value")
			w.WriteHeader(http.StatusOK)
			io.WriteString(w, responseBody)
			w.Header().Set("X-Server-Trailer", "server-trailer-value")
		},
		optQuiet,
	)

	tr := newTransport(t)

	const (
		workers    = 8
		iterations = 25
	)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				body := strings.NewReader("client request body content")
				req, err := http.NewRequest("POST", ts.URL, body)
				if err != nil {
					t.Errorf("NewRequest: %v", err)
					return
				}
				req.Header.Set("X-Test-Header", "client-header-value")
				req.Trailer = http.Header{
					"X-Client-Trailer": []string{"client-trailer-value"},
				}
				res, err := tr.RoundTrip(req)
				if err != nil {
					t.Errorf("RoundTrip: %v", err)
					return
				}
				_ = touchHeader(res.Header)
				if _, err := io.Copy(io.Discard, res.Body); err != nil {
					res.Body.Close()
					t.Errorf("client body copy: %v", err)
					return
				}
				if err := res.Body.Close(); err != nil {
					t.Errorf("body close: %v", err)
					return
				}
				_ = touchHeader(res.Trailer)
			}
		}()
	}
	wg.Wait()
}
