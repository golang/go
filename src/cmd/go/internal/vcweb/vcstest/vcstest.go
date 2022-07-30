// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package vcstest serves the repository scripts in cmd/go/testdata/vcstest
// using the [vcweb] script engine.
package vcstest

import (
	"cmd/go/internal/vcs"
	"cmd/go/internal/vcweb"
	"fmt"
	"internal/testenv"
	"io"
	"log"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

var Hosts = []string{
	"vcs-test.golang.org",
}

type Server struct {
	workDir string
	HTTP    *httptest.Server
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

	srvHTTP := httptest.NewServer(handler)

	srv = &Server{
		workDir: workDir,
		HTTP:    srvHTTP,
	}
	vcs.VCSTestRepoURL = srv.HTTP.URL
	vcs.VCSTestHosts = Hosts

	fmt.Fprintln(os.Stderr, "vcs-test.golang.org rerouted to "+srv.HTTP.URL)

	return srv, nil
}

func (srv *Server) Close() error {
	if vcs.VCSTestRepoURL != srv.HTTP.URL {
		panic("vcs URL hooks modified before Close")
	}
	vcs.VCSTestRepoURL = ""
	vcs.VCSTestHosts = nil

	srv.HTTP.Close()
	return os.RemoveAll(srv.workDir)
}
