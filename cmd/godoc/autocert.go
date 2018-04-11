// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build autocert

// This file adds automatic TLS certificate support (using
// golang.org/x/crypto/acme/autocert), conditional on the use of the
// autocert build tag. It sets the serveAutoCertHook func variable
// non-nil. It is used by main.go.
//
// TODO: make this the default? We're in the Go 1.8 freeze now, so
// this is too invasive to be default, but we want it for
// https://beta.golang.org/

package main

import (
	"crypto/tls"
	"flag"
	"net"
	"net/http"
	"time"

	"golang.org/x/crypto/acme/autocert"
	"golang.org/x/net/http2"
)

var (
	autoCertDirFlag  = flag.String("autocert_cache_dir", "/var/cache/autocert", "Directory to cache TLS certs")
	autoCertHostFlag = flag.String("autocert_hostname", "", "optional hostname to require in autocert SNI requests")
)

func init() {
	runHTTPS = runHTTPSAutocert
	certInit = certInitAutocert
	wrapHTTPMux = wrapHTTPMuxAutocert
}

var autocertManager *autocert.Manager

func certInitAutocert() {
	autocertManager = &autocert.Manager{
		Cache:  autocert.DirCache(*autoCertDirFlag),
		Prompt: autocert.AcceptTOS,
	}
	if *autoCertHostFlag != "" {
		autocertManager.HostPolicy = autocert.HostWhitelist(*autoCertHostFlag)
	}
}

func runHTTPSAutocert(h http.Handler) error {
	srv := &http.Server{
		Handler: h,
		TLSConfig: &tls.Config{
			GetCertificate: autocertManager.GetCertificate,
		},
		IdleTimeout: 60 * time.Second,
	}
	http2.ConfigureServer(srv, &http2.Server{})
	ln, err := net.Listen("tcp", ":443")
	if err != nil {
		return err
	}
	return srv.Serve(tls.NewListener(tcpKeepAliveListener{ln.(*net.TCPListener)}, srv.TLSConfig))
}

func wrapHTTPMuxAutocert(h http.Handler) http.Handler {
	return autocertManager.HTTPHandler(h)
}

// tcpKeepAliveListener sets TCP keep-alive timeouts on accepted
// connections. It's used by ListenAndServe and ListenAndServeTLS so
// dead TCP connections (e.g. closing laptop mid-download) eventually
// go away.
type tcpKeepAliveListener struct {
	*net.TCPListener
}

func (ln tcpKeepAliveListener) Accept() (c net.Conn, err error) {
	tc, err := ln.AcceptTCP()
	if err != nil {
		return
	}
	tc.SetKeepAlive(true)
	tc.SetKeepAlivePeriod(3 * time.Minute)
	return tc, nil
}
