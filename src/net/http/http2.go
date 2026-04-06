// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !nethttpomithttp2

package http

import (
	"context"
	"crypto/tls"
	"errors"
	"io"
	"log"
	"net"
	"net/http/internal/http2"
	"time"

	_ "unsafe" // for go:linkname
)

// net/http supports HTTP/2 by default, but this support is removed when
// the nethttpomithttp2 build tag is set.
//
// HTTP/2 support is provided by the net/http/internal/http2 package.
//
// This file (http2.go) connects net/http to the http2 package.
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

type http2Server = http2.Server
type http2Transport = http2.Transport

func (s *Server) configureHTTP2() {
	h2srv := &http2.Server{}

	// Historically, we've configured the HTTP/2 idle timeout in this fashion:
	// Set once at configuration time.
	if s.IdleTimeout != 0 {
		h2srv.IdleTimeout = s.IdleTimeout
	} else {
		h2srv.IdleTimeout = s.ReadTimeout
	}

	if s.TLSConfig == nil {
		s.TLSConfig = &tls.Config{}
	}
	s.nextProtoErr = h2srv.Configure(http2ServerConfig{s}, s.TLSConfig)
	if s.nextProtoErr != nil {
		return
	}

	s.RegisterOnShutdown(h2srv.GracefulShutdown)

	if s.TLSNextProto == nil {
		s.TLSNextProto = make(map[string]func(*Server, *tls.Conn, Handler))
	}
	// Historically, the presence of a TLSNextProto["h2"] key has been the signal to
	// enable/disable HTTP/2 support. Set a value in the map, but we'll never use it.
	s.TLSNextProto["h2"] = func(hs *Server, c *tls.Conn, h Handler) {
		c.Close()
	}

	s.h2 = h2srv
}

func (s *Server) serveHTTP2Conn(ctx context.Context, nc net.Conn, h Handler, sawClientPreface bool) {
	s.h2.ServeConn(nc, &http2.ServeConnOpts{
		Context:          ctx,
		Handler:          http2Handler{h},
		BaseConfig:       http2ServerConfig{s},
		SawClientPreface: sawClientPreface,
	})
}

type http2Handler struct {
	h Handler
}

func (h http2Handler) ServeHTTP(w *http2.ResponseWriter, req *http2.ServerRequest) {
	h.h.ServeHTTP(http2ResponseWriter{w}, &Request{
		ctx:           req.Context,
		Proto:         "HTTP/2.0",
		ProtoMajor:    2,
		ProtoMinor:    0,
		Method:        req.Method,
		URL:           req.URL,
		Header:        Header(req.Header),
		RequestURI:    req.RequestURI,
		Trailer:       Header(req.Trailer),
		Body:          req.Body,
		Host:          req.Host,
		ContentLength: req.ContentLength,
		RemoteAddr:    req.RemoteAddr,
		TLS:           req.TLS,
		MultipartForm: req.MultipartForm,
	})
}

type http2ResponseWriter struct {
	*http2.ResponseWriter
}

// Optional http.ResponseWriter interfaces implemented.
var (
	_ CloseNotifier   = http2ResponseWriter{}
	_ Flusher         = http2ResponseWriter{}
	_ io.StringWriter = http2ResponseWriter{}
)

func (w http2ResponseWriter) Flush()            { w.ResponseWriter.FlushError() }
func (w http2ResponseWriter) FlushError() error { return w.ResponseWriter.FlushError() }

func (w http2ResponseWriter) Header() Header { return Header(w.ResponseWriter.Header()) }

func (w http2ResponseWriter) Push(target string, opts *PushOptions) error {
	var (
		method string
		header http2.Header
	)
	if opts != nil {
		method = opts.Method
		header = http2.Header(opts.Header)
	}
	err := w.ResponseWriter.Push(target, method, header)
	if err == http2.ErrNotSupported {
		err = ErrNotSupported
	}
	return err
}

type http2ServerConfig struct {
	s *Server
}

func (s http2ServerConfig) MaxHeaderBytes() int { return s.s.MaxHeaderBytes }
func (s http2ServerConfig) ConnState(c net.Conn, st http2.ConnState) {
	if s.s.ConnState != nil {
		s.s.ConnState(c, ConnState(st))
	}
}
func (s http2ServerConfig) DoKeepAlives() bool             { return s.s.doKeepAlives() }
func (s http2ServerConfig) WriteTimeout() time.Duration    { return s.s.WriteTimeout }
func (s http2ServerConfig) SendPingTimeout() time.Duration { return s.s.ReadTimeout }
func (s http2ServerConfig) ErrorLog() *log.Logger          { return s.s.ErrorLog }
func (s http2ServerConfig) IdleTimeout() time.Duration     { return s.s.IdleTimeout }
func (s http2ServerConfig) ReadTimeout() time.Duration     { return s.s.ReadTimeout }
func (s http2ServerConfig) DisableClientPriority() bool    { return s.s.DisableClientPriority }
func (s http2ServerConfig) HTTP2Config() http2.Config {
	if s.s.HTTP2 == nil {
		return http2.Config{}
	}
	return (http2.Config)(*s.s.HTTP2)
}

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
	t2.AllowHTTP = true
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

	// Auto-configure the http2.Transport's MaxHeaderListSize from
	// the http.Transport's MaxResponseHeaderBytes. They don't
	// exactly mean the same thing, but they're close.
	if limit1 := t.MaxResponseHeaderBytes; limit1 != 0 && t2.MaxHeaderListSize == 0 {
		const h2max = 1<<32 - 1
		if limit1 >= h2max {
			t2.MaxHeaderListSize = h2max
		} else {
			t2.MaxHeaderListSize = uint32(limit1)
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
	if t.t.h2Config == nil {
		return (http2.Config)(*t.t.HTTP2)
	}
	c := (http2.Config)(*t.t.HTTP2)
	c2 := t.t.h2Config.HTTP2Config()
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

// transportFromH1Transport provides a way for HTTP/2 tests to extract
// the http2.Transport from an http.Transport.
//
//go:linkname transportFromH1Transport net/http/internal/http2_test.transportFromH1Transport
func transportFromH1Transport(t *Transport) any {
	t.nextProtoOnce.Do(t.onceSetNextProtoDefaults)
	return t.h2Transport
}
