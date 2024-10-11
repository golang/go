// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package vcstest serves the repository scripts in cmd/go/testdata/vcstest
// using the [vcweb] script engine.
package vcstest

import (
	"cmd/go/internal/vcs"
	"cmd/go/internal/vcweb"
	"cmd/go/internal/web"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"internal/testenv"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"testing"
)

var Hosts = []string{
	"vcs-test.golang.org",
}

type Server struct {
	vcweb   *vcweb.Server
	workDir string
	HTTP    *httptest.Server
	HTTPS   *httptest.Server
}

// NewServer returns a new test-local vcweb server that serves VCS requests
// for modules with paths that begin with "vcs-test.golang.org" using the
// scripts in cmd/go/testdata/vcstest.
func NewServer() (srv *Server, err error) {
	if vcs.VCSTestRepoURL != "" {
		panic("vcs URL hooks already set")
	}

	scriptDir := filepath.Join(testenv.GOROOT(nil), "src/cmd/go/testdata/vcstest")

	workDir, err := os.MkdirTemp("", "vcstest")
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			os.RemoveAll(workDir)
		}
	}()

	logger := log.Default()
	if !testing.Verbose() {
		logger = log.New(io.Discard, "", log.LstdFlags)
	}
	handler, err := vcweb.NewServer(scriptDir, workDir, logger)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			handler.Close()
		}
	}()

	srvHTTP := httptest.NewServer(handler)
	httpURL, err := url.Parse(srvHTTP.URL)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			srvHTTP.Close()
		}
	}()

	srvHTTPS := httptest.NewTLSServer(handler)
	httpsURL, err := url.Parse(srvHTTPS.URL)
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			srvHTTPS.Close()
		}
	}()

	srv = &Server{
		vcweb:   handler,
		workDir: workDir,
		HTTP:    srvHTTP,
		HTTPS:   srvHTTPS,
	}
	vcs.VCSTestRepoURL = srv.HTTP.URL
	vcs.VCSTestHosts = Hosts

	interceptors := make([]web.Interceptor, 0, 2*len(Hosts))
	for _, host := range Hosts {
		interceptors = append(interceptors,
			web.Interceptor{Scheme: "http", FromHost: host, ToHost: httpURL.Host, Client: srv.HTTP.Client()},
			web.Interceptor{Scheme: "https", FromHost: host, ToHost: httpsURL.Host, Client: srv.HTTPS.Client()})
	}
	web.EnableTestHooks(interceptors)

	fmt.Fprintln(os.Stderr, "vcs-test.golang.org rerouted to "+srv.HTTP.URL)
	fmt.Fprintln(os.Stderr, "https://vcs-test.golang.org rerouted to "+srv.HTTPS.URL)

	return srv, nil
}

func (srv *Server) Close() error {
	if vcs.VCSTestRepoURL != srv.HTTP.URL {
		panic("vcs URL hooks modified before Close")
	}
	vcs.VCSTestRepoURL = ""
	vcs.VCSTestHosts = nil
	web.DisableTestHooks()

	srv.HTTP.Close()
	srv.HTTPS.Close()
	err := srv.vcweb.Close()
	if rmErr := os.RemoveAll(srv.workDir); err == nil {
		err = rmErr
	}
	return err
}

func (srv *Server) WriteCertificateFile() (string, error) {
	b := pem.EncodeToMemory(&pem.Block{
		Type:  "CERTIFICATE",
		Bytes: srv.HTTPS.Certificate().Raw,
	})

	filename := filepath.Join(srv.workDir, "cert.pem")
	if err := os.WriteFile(filename, b, 0644); err != nil {
		return "", err
	}
	return filename, nil
}

// TLSClient returns an http.Client that can talk to the httptest.Server
// whose certificate is written to the given file path.
func TLSClient(certFile string) (*http.Client, error) {
	client := &http.Client{
		Transport: http.DefaultTransport.(*http.Transport).Clone(),
	}

	pemBytes, err := os.ReadFile(certFile)
	if err != nil {
		return nil, err
	}

	certpool := x509.NewCertPool()
	if !certpool.AppendCertsFromPEM(pemBytes) {
		return nil, fmt.Errorf("no certificates found in %s", certFile)
	}
	client.Transport.(*http.Transport).TLSClientConfig = &tls.Config{
		RootCAs: certpool,
	}

	return client, nil
}
