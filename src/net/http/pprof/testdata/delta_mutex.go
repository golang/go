// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This binary collects a 1s delta mutex profile and dumps it to os.Stdout.
//
// This is in a subprocess because we want the base mutex profile to be empty
// (as a regression test for https://go.dev/issue/64566) and the only way to
// force reset the profile is to create a new subprocess.
//
// This manually collects the HTTP response and dumps to stdout in order to
// avoid any flakiness around port selection for a real HTTP server.
package main

import (
	"bytes"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"
	"net/http/pprof"
	"runtime"
)

func main() {
	// Disable the mutex profiler. This is the default, but that default is
	// load-bearing for this test, which needs the base profile to be empty.
	runtime.SetMutexProfileFraction(0)

	h := pprof.Handler("mutex")

	req := httptest.NewRequest("GET", "/debug/pprof/mutex?seconds=1", nil)
	rec := httptest.NewRecorder()
	rec.Body = new(bytes.Buffer)

	h.ServeHTTP(rec, req)
	resp := rec.Result()
	if resp.StatusCode != http.StatusOK {
		log.Fatalf("Request failed: %s\n%s", resp.Status, rec.Body)
	}

	fmt.Print(rec.Body)
}
