// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !nethttpomithttp2 && !nethttpomithttp2client

package http

import (
	"context"
	"crypto/tls"
	"errors"
	"net"
	"net/http/internal/http2"
	"time"

	_ "unsafe" // for go:linkname
)

// This file connects the net/http Transport to the HTTP/2 client
// in the net/http/internal/http2 package. It is omitted when
// the nethttpomithttp2 or nethttpomithttp2client build tag is set.

const omitHTTP2Client = false

type http2Transport = http2.Transport

// http2ExternalTransportConfig is an HTTP/2 configuration provided by x/net/http2.
//
// When a x/net/http2.Transport wraps a net/http.Transport, we need to support the user
// setting configuration settings on the x/net Transport:
//
//	tr1 := &http.Transport{}
//	tr2 := http2.ConfigureTransports(t1)
//
//	// This setting needs to affect tr1:
//	tr2.MaxHeaderListSize = 10000
//
// We handle this by having http2.ConfigureTransports pass us an http2ExternalTransportConfig,
// which we can use to query the current state of the http2.Transport.
type http2ExternalTransportConfig interface {
	// Various configuration settings:
	HTTP2Config() HTTP2Config
	DisableCompression() bool
	MaxHeaderListSize() int64
	IdleConnTimeout() time.Duration

	// ConnFromContext is used to pass a net.Conn to Transport.NewClientConn
	// via a context value. See Transport.http2NewClientConnFromContext.
	ConnFromContext(context.Context) net.Conn

	// DialFromContext is used to dial new connections, overriding Transport.DialContext etc.
	// This is used when the user calls x/net/http2.Transport.RoundTrip directly,
	// in which case the historical behavior is to use the http2.Transport's dial functions.
	DialFromContext(ctx context.Context, network, addr string) (net.Conn, error)

	// ExternalRoundTrip reports whether Transport.RoundTrip should call the
	// external transport's RoundTrip. This is used when x/net/http2.Transport.ConnPool
	// is set, in which case the user-provided ClientConnPool has taken responsibility
	// for picking a connection to use.
	ExternalRoundTrip() bool

	// RoundTrip performs a round trip.
	// It should only be used when ExternalRoundTrip requests it.
	RoundTrip(*Request) (*Response, error)

	// Registered is called to report successful registration of the config.
	Registered(*Transport)
}

func (t *Transport) configureHTTP2(protocols Protocols) {
	if t.TLSClientConfig == nil {
		t.TLSClientConfig = &tls.Config{}
	}
	if t.HTTP2 == nil {
		t.HTTP2 = &HTTP2Config{}
	}
	t2 := http2.NewTransport(transportConfig{t})
	t.h2Transport = t2

	t.registerProtocol("https", http2RoundTripper{t2, true})
	if t.TLSNextProto == nil {
		t.TLSNextProto = make(map[string]func(authority string, c *tls.Conn) RoundTripper)
	}
	// Historically, the presence of a TLSNextProto["h2"] key has been the signal to
	// enable/disable HTTP/2 support. Set a value in the map, but we'll never use it.
	t.TLSNextProto["h2"] = func(authority string, c *tls.Conn) RoundTripper {
		return http2ErringRoundTripper{
			errors.New("unexpected use of stub RoundTripper"),
		}
	}

	// Server.ServeTLS clones the tls.Config before modifying it.
	// Transport doesn't. We may want to make the two consistent some day.
	//
	// http2configureTransport will have already set NextProtos, but adjust it again
	// here to remove HTTP/1.1 if the user has disabled it.
	t.TLSClientConfig.NextProtos = adjustNextProtos(t.TLSClientConfig.NextProtos, protocols)
}

type http2ErringRoundTripper struct{ err error }

func (rt http2ErringRoundTripper) RoundTripErr() error                   { return rt.err }
func (rt http2ErringRoundTripper) RoundTrip(*Request) (*Response, error) { return nil, rt.err }

func http2RoundTrip(req *Request, rt func(*http2.ClientRequest) (*http2.ClientResponse, error)) (*Response, error) {
	resp := &Response{}
	cresp, err := rt(&http2.ClientRequest{
		Context:       req.Context(),
		Method:        req.Method,
		URL:           req.URL,
		Header:        http2.Header(req.Header),
		Trailer:       http2.Header(req.Trailer),
		Body:          req.Body,
		Host:          req.Host,
		GetBody:       req.GetBody,
		ContentLength: req.ContentLength,
		Cancel:        req.Cancel,
		Close:         req.Close,
		ResTrailer:    (*http2.Header)(&resp.Trailer),
	})
	if err != nil {
		return nil, err
	}
	resp.Status = cresp.Status + " " + StatusText(cresp.StatusCode)
	resp.StatusCode = cresp.StatusCode
	resp.Proto = "HTTP/2.0"
	resp.ProtoMajor = 2
	resp.ProtoMinor = 0
	resp.ContentLength = cresp.ContentLength
	resp.Uncompressed = cresp.Uncompressed
	resp.Header = Header(cresp.Header)
	resp.Trailer = Header(cresp.Trailer)
	resp.Body = cresp.Body
	resp.TLS = cresp.TLS
	resp.Request = req
	return resp, nil
}

// http2AddConn adds nc to the HTTP/2 connection pool.
func (t *Transport) http2AddConn(scheme, authority string, nc net.Conn) (RoundTripper, error) {
	if t.h2Transport == nil {
		return nil, errors.ErrUnsupported
	}
	err := t.h2Transport.AddConn(scheme, authority, nc)
	if err != nil {
		return nil, err
	}
	return http2RoundTripper{t.h2Transport, false}, nil
}

// http2NewClientConn creates an HTTP/2 genericClientConn (used to implement ClientConn) from nc.
// The connection is not added to the HTTP/2 connection pool.
func (t *Transport) http2NewClientConn(nc net.Conn, internalStateHook func()) (genericClientConn, error) {
	if t.h2Transport == nil {
		return nil, errors.ErrUnsupported
	}
	cc, err := t.h2Transport.NewClientConn(nc, internalStateHook)
	if err != nil {
		return nil, err
	}
	return http2ClientConn{cc}, nil
}

// http2NewClientConnFromContext creates a *ClientConn from a net.Conn.
//
// Transport.NewClientConn takes an address and dials a new net.Conn.
// We don't currently provide a simple way for the user to provide a net.Conn and get a
// *ClientConn out of it (although we do let the user provide their own Transport.DialContext,
// which can be used to effectively do this).
//
// x/net/http2.Transport.NewClientConn, in contrast, requires the user to provide a net.Conn.
// To support implementing the x/net/http2 NewClientConn in terms of a net/http.Transport,
// we permit x/net/http2 to pass us a net.Conn via a context key.
//
// http2NewClientConnFromContext handles extracting the net.Conn from the Context
// (when present) and creating a *ClientConn from it.
func (t *Transport) http2NewClientConnFromContext(ctx context.Context) (*ClientConn, error) {
	if t.h2Config == nil {
		return nil, errors.ErrUnsupported
	}
	nc := t.h2Config.ConnFromContext(ctx)
	if nc == nil {
		return nil, errors.ErrUnsupported
	}
	if t.h2Transport == nil {
		return nil, errors.New("http: Transport does not support HTTP/2")
	}
	cc := &ClientConn{}
	gc, err := t.http2NewClientConn(nc, cc.maybeRunStateHook)
	if err != nil {
		return nil, err
	}
	cc.stateHookMu.Lock()
	defer cc.stateHookMu.Unlock()
	cc.cc = gc
	cc.lastAvailable = gc.Available()
	return cc, nil
}

// http2ExternalDial creates a new HTTP/2 connection,
// using the x/net/http2.Transport's dial functions.
//
// This is used when the user has called x/net/http2.Transport.RoundTrip.
// If the RoundTrip needs to create a new connection,
// the historical behavior is for it to use the http2.Transport's DialTLS or DialTLSContext
// functions, and not any dial functions on the http.Transport.
func (t *Transport) http2ExternalDial(ctx context.Context, cm connectMethod) (RoundTripper, error) {
	if t.h2Config == nil {
		return nil, errors.ErrUnsupported
	}
	nc, err := t.h2Config.DialFromContext(ctx, "tcp", cm.targetAddr)
	if err != nil {
		return nil, err
	}
	return t.http2AddConn(cm.targetScheme, cm.targetAddr, nc)
}

type http2RoundTripper struct {
	t                *http2.Transport
	mapCachedConnErr bool
}

func (rt http2RoundTripper) RoundTrip(req *Request) (*Response, error) {
	resp, err := http2RoundTrip(req, rt.t.RoundTrip)
	if err != nil {
		if rt.mapCachedConnErr && http2isNoCachedConnError(err) {
			err = ErrSkipAltProtocol
		}
		return nil, err
	}
	return resp, nil
}

type http2ClientConn struct {
	http2.NetHTTPClientConn
}

func (cc http2ClientConn) RoundTrip(req *Request) (*Response, error) {
	return http2RoundTrip(req, cc.NetHTTPClientConn.RoundTrip)
}

// transportConfig implements the http2.TransportConfig interface,
// providing the net/http Transport's configuration to the HTTP/2 implementation.
//
// When an x/net/http2 Transport has provided a configuration (see http2ExternalTransportConfig),
// the transportConfig merges the x/net/http2 and net/http Transport configurations.
type transportConfig struct {
	t *Transport
}

func (t transportConfig) MaxResponseHeaderBytes() int64        { return t.t.MaxResponseHeaderBytes }
func (t transportConfig) DisableKeepAlives() bool              { return t.t.DisableKeepAlives }
func (t transportConfig) ExpectContinueTimeout() time.Duration { return t.t.ExpectContinueTimeout }
func (t transportConfig) ResponseHeaderTimeout() time.Duration { return t.t.ResponseHeaderTimeout }

func (t transportConfig) MaxHeaderListSize() int64 {
	if t.t.h2Config != nil {
		return t.t.h2Config.MaxHeaderListSize()
	}
	return 0
}

func (t transportConfig) DisableCompression() bool {
	if t.t.h2Config != nil && t.t.h2Config.DisableCompression() {
		return true
	}
	return t.t.DisableCompression
}

func (t transportConfig) IdleConnTimeout() time.Duration {
	// Unlike most config settings, historically IdleConnTimeout prefers the
	// http2.Transport's setting over the http.Transport.
	if t.t.h2Config != nil {
		if timeout := t.t.h2Config.IdleConnTimeout(); timeout != 0 {
			return timeout
		}
	}
	return t.t.IdleConnTimeout
}

func (t transportConfig) HTTP2Config() http2.Config {
	return mergeHTTP2Config(t.t.HTTP2, t.t.h2Config)
}

// transportFromH1Transport provides a way for HTTP/2 tests to extract
// the http2.Transport from an http.Transport.
//
//go:linkname transportFromH1Transport net/http/internal/http2_test.transportFromH1Transport
func transportFromH1Transport(t *Transport) any {
	t.nextProtoOnce.Do(t.onceSetNextProtoDefaults)
	return t.h2Transport
}
