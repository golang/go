// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm
// +build js,wasm

package http

import (
	"time"
)

// DefaultTransport is the default implementation of Transport and is
// used by DefaultClient. It uses HTTP proxies as directed by the
// $HTTP_PROXY and $NO_PROXY (or $http_proxy and $no_proxy) environment
// variables. No default dialer is specified as the Fetch API will be
// used in RoundTrip if not specified.
var DefaultTransport RoundTripper = &Transport{
	Proxy:                 ProxyFromEnvironment,
	ForceAttemptHTTP2:     true,
	MaxIdleConns:          100,
	IdleConnTimeout:       90 * time.Second,
	TLSHandshakeTimeout:   10 * time.Second,
	ExpectContinueTimeout: 1 * time.Second,
}
