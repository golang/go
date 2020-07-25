// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path"
	"sync"

	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func (s *Server) initialize(ctx context.Context, params *protocol.ParamInitialize) (*protocol.InitializeResult, error) {
	s.stateMu.Lock()
	if s.state >= serverInitializing {
		defer s.stateMu.Unlock()
		return nil, fmt.Errorf("%w: initialize called while server in %v state", jsonrpc2.ErrInvalidRequest, s.state)
	}
	s.state = serverInitializing
	s.stateMu.Unlock()

	s.supportsWorkDoneProgress = params.Capabilities.Window.WorkDoneProgress
	s.inProgress = map[string]*WorkDone{}

	options := s.session.Options()
	defer func() { s.session.SetOptions(options) }()

	// TODO: Handle results here.
	source.SetOptions(&options, params.InitializationOptions)
	options.ForClientCapabilities(params.Capabilities)

	if params.RootURI != "" && !params.RootURI.SpanURI().IsFile() {
		return nil, fmt.Errorf("unsupported URI scheme: %v (gopls only supports file URIs)", params.RootURI)
	}
	s.pendingFolders = params.WorkspaceFolders
	if len(s.pendingFolders) == 0 {
		if params.RootURI != "" {
			s.pendingFolders = []protocol.WorkspaceFolder{{
				URI:  string(params.RootURI),
				Name: path.Base(params.RootURI.SpanURI().Filename()),
			}}
		}
	}

	var codeActionProvider interface{} = true
	if ca := params.Capabilities.TextDocument.CodeAction; len(ca.CodeActionLiteralSupport.CodeActionKind.ValueSet) > 0 {
		// If the client has specified CodeActionLiteralSupport,
		// send the code actions we support.
		//
		// Using CodeActionOptions is only valid if codeActionLiteralSupport is set.
		codeActionProvider = &protocol.CodeActionOptions{
			CodeActionKinds: s.getSupportedCodeActions(),
		}
	}
	var renameOpts interface{} = true
	if r := params.Capabilities.TextDocument.Rename; r.PrepareSupport {
		renameOpts = protocol.RenameOptions{
			PrepareProvider: r.PrepareSupport,
		}
	}

	goplsVer := &bytes.Buffer{}
	debug.PrintVersionInfo(ctx, goplsVer, true, debug.PlainText)

	return &protocol.InitializeResult{
		Capabilities: protocol.ServerCapabilities{
			CodeActionProvider: codeActionProvider,
			CompletionProvider: protocol.CompletionOptions{
				TriggerCharacters: []string{"."},
			},
			DefinitionProvider:         true,
			TypeDefinitionProvider:     true,
			ImplementationProvider:     true,
			DocumentFormattingProvider: true,
			DocumentSymbolProvider:     true,
			WorkspaceSymbolProvider:    true,
			ExecuteCommandProvider: protocol.ExecuteCommandOptions{
				Commands: options.SupportedCommands,
			},
			FoldingRangeProvider:      true,
			HoverProvider:             true,
			DocumentHighlightProvider: true,
			DocumentLinkProvider:      protocol.DocumentLinkOptions{},
			ReferencesProvider:        true,
			RenameProvider:            renameOpts,
			SignatureHelpProvider: protocol.SignatureHelpOptions{
				TriggerCharacters: []string{"(", ","},
			},
			TextDocumentSync: &protocol.TextDocumentSyncOptions{
				Change:    protocol.Incremental,
				OpenClose: true,
				Save: protocol.SaveOptions{
					IncludeText: false,
				},
			},
			Workspace: protocol.WorkspaceGn{
				WorkspaceFolders: protocol.WorkspaceFoldersGn{
					Supported:           true,
					ChangeNotifications: "workspace/didChangeWorkspaceFolders",
				},
			},
		},
		ServerInfo: struct {
			Name    string `json:"name"`
			Version string `json:"version,omitempty"`
		}{
			Name:    "gopls",
			Version: goplsVer.String(),
		},
	}, nil
}

func (s *Server) initialized(ctx context.Context, params *protocol.InitializedParams) error {
	s.stateMu.Lock()
	if s.state >= serverInitialized {
		defer s.stateMu.Unlock()
		return fmt.Errorf("%w: initalized called while server in %v state", jsonrpc2.ErrInvalidRequest, s.state)
	}
	s.state = serverInitialized
	s.stateMu.Unlock()

	options := s.session.Options()
	defer func() { s.session.SetOptions(options) }()

	var registrations []protocol.Registration
	if options.ConfigurationSupported && options.DynamicConfigurationSupported {
		registrations = append(registrations,
			protocol.Registration{
				ID:     "workspace/didChangeConfiguration",
				Method: "workspace/didChangeConfiguration",
			},
			protocol.Registration{
				ID:     "workspace/didChangeWorkspaceFolders",
				Method: "workspace/didChangeWorkspaceFolders",
			},
		)
	}

	// TODO: this event logging may be unnecessary.
	// The version info is included in the initialize response.
	buf := &bytes.Buffer{}
	debug.PrintVersionInfo(ctx, buf, true, debug.PlainText)
	event.Log(ctx, buf.String())

	if err := s.addFolders(ctx, s.pendingFolders); err != nil {
		return err
	}
	s.pendingFolders = nil

	if options.DynamicWatchedFilesSupported {
		for _, view := range s.session.Views() {
			dirs, err := view.WorkspaceDirectories(ctx)
			if err != nil {
				return err
			}
			for _, dir := range dirs {
				registrations = append(registrations, protocol.Registration{
					ID:     "workspace/didChangeWatchedFiles",
					Method: "workspace/didChangeWatchedFiles",
					RegisterOptions: protocol.DidChangeWatchedFilesRegistrationOptions{
						Watchers: []protocol.FileSystemWatcher{{
							GlobPattern: fmt.Sprintf("%s/**/*.{go,mod,sum}", dir),
							Kind:        float64(protocol.WatchChange + protocol.WatchDelete + protocol.WatchCreate),
						}},
					},
				})
			}
		}
		if len(registrations) > 0 {
			s.client.RegisterCapability(ctx, &protocol.RegistrationParams{
				Registrations: registrations,
			})
		}
	}
	return nil
}

func (s *Server) addFolders(ctx context.Context, folders []protocol.WorkspaceFolder) error {
	originalViews := len(s.session.Views())
	viewErrors := make(map[span.URI]error)

	var wg sync.WaitGroup
	if s.session.Options().VerboseWorkDoneProgress {
		work := s.StartWork(ctx, DiagnosticWorkTitle(FromInitialWorkspaceLoad), "Calculating diagnostics for initial workspace load...", nil)
		defer func() {
			go func() {
				wg.Wait()
				work.End(ctx, "Done.")
			}()
		}()
	}
	for _, folder := range folders {
		uri := span.URIFromURI(folder.URI)
		view, snapshot, err := s.addView(ctx, folder.Name, uri)
		if err != nil {
			viewErrors[uri] = err
			continue
		}
		// Print each view's environment.
		buf := &bytes.Buffer{}
		if err := view.WriteEnv(ctx, buf); err != nil {
			event.Error(ctx, "failed to write environment", err, tag.Directory.Of(view.Folder().Filename()))
			continue
		}
		event.Log(ctx, buf.String())

		// Diagnose the newly created view.
		wg.Add(1)
		go func() {
			s.diagnoseDetached(snapshot)
			wg.Done()
		}()
	}
	if len(viewErrors) > 0 {
		errMsg := fmt.Sprintf("Error loading workspace folders (expected %v, got %v)\n", len(folders), len(s.session.Views())-originalViews)
		for uri, err := range viewErrors {
			errMsg += fmt.Sprintf("failed to load view for %s: %v\n", uri, err)
		}
		return s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
			Type:    protocol.Error,
			Message: errMsg,
		})
	}
	return nil
}

func (s *Server) fetchConfig(ctx context.Context, name string, folder span.URI, o *source.Options) error {
	if !s.session.Options().ConfigurationSupported {
		return nil
	}
	v := protocol.ParamConfiguration{
		ConfigurationParams: protocol.ConfigurationParams{
			Items: []protocol.ConfigurationItem{{
				ScopeURI: string(folder),
				Section:  "gopls",
			}, {
				ScopeURI: string(folder),
				Section:  fmt.Sprintf("gopls-%s", name),
			}},
		},
	}
	configs, err := s.client.Configuration(ctx, &v)
	if err != nil {
		return fmt.Errorf("failed to get workspace configuration from client (%s): %v", folder, err)
	}
	for _, config := range configs {
		results := source.SetOptions(o, config)
		for _, result := range results {
			if result.Error != nil {
				if err := s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
					Type:    protocol.Error,
					Message: result.Error.Error(),
				}); err != nil {
					return err
				}
			}
			switch result.State {
			case source.OptionUnexpected:
				if err := s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
					Type:    protocol.Error,
					Message: fmt.Sprintf("unexpected config %s", result.Name),
				}); err != nil {
					return err
				}
			case source.OptionDeprecated:
				msg := fmt.Sprintf("config %s is deprecated", result.Name)
				if result.Replacement != "" {
					msg = fmt.Sprintf("%s, use %s instead", msg, result.Replacement)
				}
				if err := s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
					Type:    protocol.Warning,
					Message: msg,
				}); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// beginFileRequest checks preconditions for a file-oriented request and routes
// it to a snapshot.
// We don't want to return errors for benign conditions like wrong file type,
// so callers should do if !ok { return err } rather than if err != nil.
func (s *Server) beginFileRequest(ctx context.Context, pURI protocol.DocumentURI, expectKind source.FileKind) (source.Snapshot, source.FileHandle, bool, error) {
	uri := pURI.SpanURI()
	if !uri.IsFile() {
		// Not a file URI. Stop processing the request, but don't return an error.
		return nil, nil, false, nil
	}
	view, err := s.session.ViewOf(uri)
	if err != nil {
		return nil, nil, false, err
	}
	snapshot := view.Snapshot()
	fh, err := snapshot.GetFile(ctx, uri)
	if err != nil {
		return nil, nil, false, err
	}
	if expectKind != source.UnknownKind && fh.Kind() != expectKind {
		// Wrong kind of file. Nothing to do.
		return nil, nil, false, nil
	}
	return snapshot, fh, true, nil
}

func (s *Server) shutdown(ctx context.Context) error {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	if s.state < serverInitialized {
		event.Log(ctx, "server shutdown without initialization")
	}
	if s.state != serverShutDown {
		// drop all the active views
		s.session.Shutdown(ctx)
		s.state = serverShutDown
	}
	return nil
}

func (s *Server) exit(ctx context.Context) error {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()

	// TODO: We need a better way to find the conn close method.
	s.client.(io.Closer).Close()

	if s.state != serverShutDown {
		// TODO: We should be able to do better than this.
		os.Exit(1)
	}
	// we don't terminate the process on a normal exit, we just allow it to
	// close naturally if needed after the connection is closed.
	return nil
}
