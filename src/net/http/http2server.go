// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !nethttpomithttp2 && !nethttpomithttp2server

package http

import (
	"context"
	"crypto/tls"
	"io"
	"log"
	"net"
	"net/http/internal/http2"
	"time"
)

// This file connects the net/http Server to the HTTP/2 server
// in the net/http/internal/http2 package. It is omitted when
// the nethttpomithttp2 or nethttpomithttp2server build tag is set.

const omitHTTP2Server = false

type http2Server = http2.Server

func (s *Server) configureHTTP2() {
	h2srv := &http2.Server{}

	// Historically, we've configured the HTTP/2 idle timeout in this fashion:
	// Set once at configuration time.
	if s.IdleTimeout != 0 {
		s.h2IdleTimeout = s.IdleTimeout
	} else {
		s.h2IdleTimeout = s.ReadTimeout
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

func (s *Server) setHTTP2Config(conf http2ExternalServerConfig) {
	if s.h2Config != nil {
		panic("http: HTTP/2 Server already registered")
	}
	s.h2Config = conf
	s.h2Config.ServeConnFunc(func(ctx context.Context, nc net.Conn, h Handler, sawClientPreface bool, upgradeReq *Request, settings []byte) {
		s.serveHTTP2Conn(ctx, nc, h, sawClientPreface, upgradeReq, settings, nil)
	})
	s.configureHTTP2()
}

// serveHTTP2Conn serves nc with the HTTP/2 server. It may return before the
// connection is done being served (an idle HTTP/2 connection doesn't hold
// onto a goroutine); onClose, if non-nil, runs once the connection is done
// and has been closed.
func (s *Server) serveHTTP2Conn(ctx context.Context, nc net.Conn, h Handler, sawClientPreface bool, upgradeReq *Request, settings []byte, onClose func()) {
	s.setupHTTP2_ServeTLS()
	var serverUpgradeReq *http2.ServerRequest
	if upgradeReq != nil {
		serverUpgradeReq = http2ServerRequestFromRequest(upgradeReq)
	}
	nc.SetReadDeadline(time.Time{})
	nc.SetWriteDeadline(time.Time{})
	s.h2.ServeConn(nc, &http2.ServeConnOpts{
		Context:          ctx,
		Handler:          http2Handler{h},
		BaseConfig:       http2ServerConfig{s},
		SawClientPreface: sawClientPreface,
		UpgradeRequest:   serverUpgradeReq,
		Settings:         settings,
		OnClose:          onClose,
	})
}

func http2ServerRequestFromRequest(req *Request) *http2.ServerRequest {
	return &http2.ServerRequest{
		Context:       req.Context(),
		Proto:         req.Proto,
		ProtoMajor:    req.ProtoMajor,
		ProtoMinor:    req.ProtoMinor,
		Method:        req.Method,
		URL:           req.URL,
		Header:        http2.Header(req.Header),
		Trailer:       http2.Header(req.Trailer),
		Body:          req.Body,
		Host:          req.Host,
		ContentLength: req.ContentLength,
		RemoteAddr:    req.RemoteAddr,
		RequestURI:    req.RequestURI,
		TLS:           req.TLS,
	}
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

func (s http2ServerConfig) MaxHeaderBytes() int      { return s.s.MaxHeaderBytes }
func (s http2ServerConfig) MaxHeaderValueCount() int { return s.s.maxHeaderValueCount() }
func (s http2ServerConfig) ConnState(c net.Conn, st http2.ConnState) {
	if s.s.ConnState != nil {
		s.s.ConnState(c, ConnState(st))
	}
}
func (s http2ServerConfig) DoKeepAlives() bool             { return s.s.doKeepAlives() }
func (s http2ServerConfig) WriteTimeout() time.Duration    { return s.s.WriteTimeout }
func (s http2ServerConfig) SendPingTimeout() time.Duration { return s.s.ReadTimeout }
func (s http2ServerConfig) ErrorLog() *log.Logger          { return s.s.ErrorLog }
func (s http2ServerConfig) ReadTimeout() time.Duration     { return s.s.ReadTimeout }
func (s http2ServerConfig) DisableClientPriority() bool    { return s.s.DisableClientPriority }

func (s http2ServerConfig) IdleTimeout() time.Duration {
	if s.s.h2Config != nil {
		return s.s.h2Config.IdleTimeout()
	}
	return s.s.h2IdleTimeout
}

func (s http2ServerConfig) HTTP2Config() http2.Config {
	return mergeHTTP2Config(s.s.HTTP2, s.s.h2Config)
}

// http2ExternalServerConfig is an HTTP/2 configuration provided by x/net/http2.
//
// When a x/net/http2.Server wraps a net/http.Server, we need to support the user
// setting configuration settings on the x/net Server:
//
//	s1 := &http.Server{}
//	s2 := &http2.Server{}
//	http2.ConfigureServer(s1, s2)
//
//	// This setting needs to affect s1:
//	s2.MaxReadFrameSize = 10000
//
// We handle this by having http2.ConfigureServer pass us an http2ExternalServerConfig
// (see http.Server.Serve) which we can use to query the current state of the http2.Server.
type http2ExternalServerConfig interface {
	// Various configuration settings:
	HTTP2Config() HTTP2Config
	IdleTimeout() time.Duration

	// ServeConnFunc provides a function to the x/net/http2.Server which it
	// can use to serve a new connection.
	ServeConnFunc(func(ctx context.Context, nc net.Conn, h Handler, sawClientPreface bool, upgradeReq *Request, settings []byte))
}
