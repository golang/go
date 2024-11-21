// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package intercept

import (
	"errors"
	"net/http"
	"net/url"
)

// Interceptor is used to change the host, and maybe the client,
// for a request to point to a test host.
type Interceptor struct {
	Scheme   string
	FromHost string
	ToHost   string
	Client   *http.Client
}

// EnableTestHooks installs the given interceptors to be used by URL and Request.
func EnableTestHooks(interceptors []Interceptor) error {
	if TestHooksEnabled {
		return errors.New("web: test hooks already enabled")
	}

	for _, t := range interceptors {
		if t.FromHost == "" {
			panic("EnableTestHooks: missing FromHost")
		}
		if t.ToHost == "" {
			panic("EnableTestHooks: missing ToHost")
		}
	}

	testInterceptors = interceptors
	TestHooksEnabled = true
	return nil
}

// DisableTestHooks disables the installed interceptors.
func DisableTestHooks() {
	if !TestHooksEnabled {
		panic("web: test hooks not enabled")
	}
	TestHooksEnabled = false
	testInterceptors = nil
}

var (
	// TestHooksEnabled is true if interceptors are installed
	TestHooksEnabled = false
	testInterceptors []Interceptor
)

// URL returns the Interceptor to be used for a given URL.
func URL(u *url.URL) (*Interceptor, bool) {
	if !TestHooksEnabled {
		return nil, false
	}
	for i, t := range testInterceptors {
		if u.Host == t.FromHost && (u.Scheme == "" || u.Scheme == t.Scheme) {
			return &testInterceptors[i], true
		}
	}
	return nil, false
}

// Request updates the host to actually use for the request, if it is to be intercepted.
func Request(req *http.Request) {
	if t, ok := URL(req.URL); ok {
		req.Host = req.URL.Host
		req.URL.Host = t.ToHost
	}
}
