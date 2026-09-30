// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !nethttpomithttp2 && !(nethttpomithttp2server && nethttpomithttp2client)

package http

import "net/http/internal/http2"

// net/http supports HTTP/2 by default, but this support is removed when
// the nethttpomithttp2 build tag is set. The nethttpomithttp2server and
// nethttpomithttp2client build tags remove only HTTP/2 server or client
// support, respectively. Setting both is equivalent to nethttpomithttp2.
//
// HTTP/2 support is provided by the net/http/internal/http2 package.
//
// This file (http2.go), along with http2server.go and http2client.go,
// connects net/http to the http2 package.
// Since http imports http2, to avoid an import cycle we need to
// translate http package types (e.g., Request) into the equivalent
// http2 package types (e.g., http2.ClientRequest).
//
// The golang.org/x/net/http2 package is the original source of truth for
// the HTTP/2 implementation. At this time, users may still import that
// package and register its implementation on a net/http Transport or Server.
// However, the x/net package is no longer synchronized with std.

func init() {
	// NoBody and LocalAddrContextKey need to have the same value
	// in the http and http2 packages.
	//
	// We can't define these values in net/http/internal,
	// because their concrete types are part of the net/http API and
	// moving them causes API checker failures.
	// Override the http2 package versions at init time instead.
	http2.LocalAddrContextKey = LocalAddrContextKey
	http2.NoBody = NoBody
}

type http2Configer interface {
	HTTP2Config() HTTP2Config
}

func mergeHTTP2Config(c1 *HTTP2Config, confer http2Configer) http2.Config {
	if c1 == nil && confer == nil {
		return http2.Config{}
	}
	var c http2.Config
	if c1 != nil {
		c = (http2.Config)(*c1)
	}
	var c2 HTTP2Config
	if confer != nil {
		c2 = confer.HTTP2Config()
	}
	if c.MaxConcurrentStreams == 0 {
		c.MaxConcurrentStreams = c2.MaxConcurrentStreams
	}
	if c2.StrictMaxConcurrentRequests {
		c.StrictMaxConcurrentRequests = true
	}
	if c.MaxDecoderHeaderTableSize == 0 {
		c.MaxDecoderHeaderTableSize = c2.MaxDecoderHeaderTableSize
	}
	if c.MaxEncoderHeaderTableSize == 0 {
		c.MaxEncoderHeaderTableSize = c2.MaxEncoderHeaderTableSize
	}
	if c.MaxReadFrameSize == 0 {
		c.MaxReadFrameSize = c2.MaxReadFrameSize
	}
	if c.MaxReceiveBufferPerConnection == 0 {
		c.MaxReceiveBufferPerConnection = c2.MaxReceiveBufferPerConnection
	}
	if c.MaxReceiveBufferPerStream == 0 {
		c.MaxReceiveBufferPerStream = c2.MaxReceiveBufferPerStream
	}
	if c.SendPingTimeout == 0 {
		c.SendPingTimeout = c2.SendPingTimeout
	}
	if c.PingTimeout == 0 {
		c.PingTimeout = c2.PingTimeout
	}
	if c.WriteByteTimeout == 0 {
		c.WriteByteTimeout = c2.WriteByteTimeout
	}
	if c2.PermitProhibitedCipherSuites {
		c.PermitProhibitedCipherSuites = true
	}
	if c.CountError == nil {
		c.CountError = c2.CountError
	}
	return c
}
