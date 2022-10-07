// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package lsp implements LSP for gopls.
package lsp

import (
	"context"
	"fmt"
	"sync"

	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/progress"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/jsonrpc2"
)

const concurrentAnalyses = 1

// NewServer creates an LSP server and binds it to handle incoming client
// messages on on the supplied stream.
func NewServer(session *cache.Session, client protocol.ClientCloser) *Server {
	tracker := progress.NewTracker(client)
	session.SetProgressTracker(tracker)
	return &Server{
		diagnostics:           map[span.URI]*fileReports{},
		gcOptimizationDetails: make(map[string]struct{}),
		watchedGlobPatterns:   make(map[string]struct{}),
		changedFiles:          make(map[span.URI]struct{}),
		session:               session,
		client:                client,
		diagnosticsSema:       make(chan struct{}, concurrentAnalyses),
		progress:              tracker,
		diagDebouncer:         newDebouncer(),
		watchedFileDebouncer:  newDebouncer(),
	}
}

type serverState int

const (
	serverCreated      = serverState(iota)
	serverInitializing // set once the server has received "initialize" request
	serverInitialized  // set once the server has received "initialized" request
	serverShutDown
)

func (s serverState) String() string {
	switch s {
	case serverCreated:
		return "created"
	case serverInitializing:
		return "initializing"
	case serverInitialized:
		return "initialized"
	case serverShutDown:
		return "shutDown"
	}
	return fmt.Sprintf("(unknown state: %d)", int(s))
}

// Server implements the protocol.Server interface.
type Server struct {
	client protocol.ClientCloser

	stateMu sync.Mutex
	state   serverState
	// notifications generated before serverInitialized
	notifications []*protocol.ShowMessageParams

	session *cache.Session

	tempDir string

	// changedFiles tracks files for which there has been a textDocument/didChange.
	changedFilesMu sync.Mutex
	changedFiles   map[span.URI]struct{}

	// folders is only valid between initialize and initialized, and holds the
	// set of folders to build views for when we are ready
	pendingFolders []protocol.WorkspaceFolder

	// watchedGlobPatterns is the set of glob patterns that we have requested
	// the client watch on disk. It will be updated as the set of directories
	// that the server should watch changes.
	watchedGlobPatternsMu  sync.Mutex
	watchedGlobPatterns    map[string]struct{}
	watchRegistrationCount int

	diagnosticsMu sync.Mutex
	diagnostics   map[span.URI]*fileReports

	// gcOptimizationDetails describes the packages for which we want
	// optimization details to be included in the diagnostics. The key is the
	// ID of the package.
	gcOptimizationDetailsMu sync.Mutex
	gcOptimizationDetails   map[string]struct{}

	// diagnosticsSema limits the concurrency of diagnostics runs, which can be
	// expensive.
	diagnosticsSema chan struct{}

	progress *progress.Tracker

	// diagDebouncer is used for debouncing diagnostics.
	diagDebouncer *debouncer

	// watchedFileDebouncer is used for batching didChangeWatchedFiles notifications.
	watchedFileDebouncer *debouncer
	fileChangeMu         sync.Mutex
	pendingOnDiskChanges []*pendingModificationSet

	// When the workspace fails to load, we show its status through a progress
	// report with an error message.
	criticalErrorStatusMu sync.Mutex
	criticalErrorStatus   *progress.WorkDone
}

type pendingModificationSet struct {
	diagnoseDone chan struct{}
	changes      []source.FileModification
}

func (s *Server) workDoneProgressCancel(ctx context.Context, params *protocol.WorkDoneProgressCancelParams) error {
	return s.progress.Cancel(params.Token)
}

func (s *Server) nonstandardRequest(ctx context.Context, method string, params interface{}) (interface{}, error) {
	switch method {
	case "gopls/diagnoseFiles":
		paramMap := params.(map[string]interface{})
		for _, file := range paramMap["files"].([]interface{}) {
			snapshot, fh, ok, release, err := s.beginFileRequest(ctx, protocol.DocumentURI(file.(string)), source.UnknownKind)
			defer release()
			if !ok {
				return nil, err
			}

			fileID, diagnostics, err := source.FileDiagnostics(ctx, snapshot, fh.URI())
			if err != nil {
				return nil, err
			}
			if err := s.client.PublishDiagnostics(ctx, &protocol.PublishDiagnosticsParams{
				URI:         protocol.URIFromSpanURI(fh.URI()),
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

func notImplemented(method string) error {
	return fmt.Errorf("%w: %q not yet implemented", jsonrpc2.ErrMethodNotFound, method)
}

//go:generate helper/helper -d protocol/tsserver.go -o server_gen.go -u .
