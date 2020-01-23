// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lsp implements LSP for gopls.
package lsp

import (
	"context"
	"fmt"
	"net"
	"sync"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// NewClientServer
func NewClientServer(ctx context.Context, session source.Session, client protocol.Client) (context.Context, *Server) {
	ctx = protocol.WithClient(ctx, client)
	return ctx, &Server{
		client:    client,
		session:   session,
		delivered: make(map[span.URI]sentDiagnostics),
	}
}

// NewServer creates an LSP server and binds it to handle incoming client
// messages on on the supplied stream.
func NewServer(ctx context.Context, session source.Session, stream jsonrpc2.Stream) (context.Context, *Server) {
	s := &Server{
		delivered: make(map[span.URI]sentDiagnostics),
		session:   session,
	}
	ctx, s.Conn, s.client = protocol.NewServer(ctx, stream, s)
	return ctx, s
}

// RunServerOnPort starts an LSP server on the given port and does not exit.
// This function exists for debugging purposes.
func RunServerOnPort(ctx context.Context, cache source.Cache, port int, h func(ctx context.Context, s *Server)) error {
	return RunServerOnAddress(ctx, cache, fmt.Sprintf(":%v", port), h)
}

// RunServerOnAddress starts an LSP server on the given address and does not
// exit. This function exists for debugging purposes.
func RunServerOnAddress(ctx context.Context, cache source.Cache, addr string, h func(ctx context.Context, s *Server)) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	for {
		conn, err := ln.Accept()
		if err != nil {
			return err
		}
		h(NewServer(ctx, cache.NewSession(), jsonrpc2.NewHeaderStream(conn, conn)))
	}
}

func (s *Server) Run(ctx context.Context) error {
	return s.Conn.Run(ctx)
}

type serverState int

const (
	serverCreated      = serverState(iota)
	serverInitializing // set once the server has received "initialize" request
	serverInitialized  // set once the server has received "initialized" request
	serverShutDown
)

type Server struct {
	Conn   *jsonrpc2.Conn
	client protocol.Client

	stateMu sync.Mutex
	state   serverState

	session source.Session

	// changedFiles tracks files for which there has been a textDocument/didChange.
	changedFiles map[span.URI]struct{}

	// folders is only valid between initialize and initialized, and holds the
	// set of folders to build views for when we are ready
	pendingFolders []protocol.WorkspaceFolder

	// delivered is a cache of the diagnostics that the server has sent.
	deliveredMu sync.Mutex
	delivered   map[span.URI]sentDiagnostics
}

// sentDiagnostics is used to cache diagnostics that have been sent for a given file.
type sentDiagnostics struct {
	version      float64
	identifier   string
	sorted       []source.Diagnostic
	withAnalysis bool
	snapshotID   uint64
}

func (s *Server) cancelRequest(ctx context.Context, params *protocol.CancelParams) error {
	return nil
}

func (s *Server) codeLens(ctx context.Context, params *protocol.CodeLensParams) ([]protocol.CodeLens, error) {
	return nil, nil
}

func (s *Server) nonstandardRequest(ctx context.Context, method string, params interface{}) (interface{}, error) {
	paramMap := params.(map[string]interface{})
	if method == "gopls/diagnoseFiles" {
		for _, file := range paramMap["files"].([]interface{}) {
			uri := span.URI(file.(string))
			view, err := s.session.ViewOf(uri)
			if err != nil {
				return nil, err
			}
			fileID, diagnostics, err := source.FileDiagnostics(ctx, view.Snapshot(), uri)
			if err != nil {
				return nil, err
			}
			if err := s.client.PublishDiagnostics(ctx, &protocol.PublishDiagnosticsParams{
				URI:         protocol.NewURI(uri),
				Diagnostics: toProtocolDiagnostics(diagnostics),
				Version:     fileID.Version,
			}); err != nil {
				return nil, err
			}
		}
		if err := s.client.PublishDiagnostics(ctx, &protocol.PublishDiagnosticsParams{
			URI: "gopls://diagnostics-done",
		}); err != nil {
			return nil, err
		}
		return struct{}{}, nil
	}
	return nil, notImplemented(method)
}

func notImplemented(method string) *jsonrpc2.Error {
	return jsonrpc2.NewErrorf(jsonrpc2.CodeMethodNotFound, "method %q not yet implemented", method)
}

//go:generate helper/helper -d protocol/tsserver.go -o server_gen.go -u .
