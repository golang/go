// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !js

package http

// RoundTrip implements the RoundTripper interface.
//
// For higher-level HTTP client support (such as handling of cookies
// and redirects), see Get, Post, and the Client type.
//
// Like the RoundTripper interface, the error types returned
// by RoundTrip are unspecified.
func (t *Transport) RoundTrip(req *Request) (*Response, error) {
	return t.roundTrip(req)
}
