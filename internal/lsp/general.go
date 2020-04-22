// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"path"

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

	if !params.RootURI.SpanURI().IsFile() {
		return nil, fmt.Errorf("unsupported URI scheme: %v (gopls only supports file URIs)", params.RootURI)
	}
	s.pendingFolders = params.WorkspaceFolders
	if len(s.pendingFolders) == 0 {
		if params.RootURI != "" {
			s.pendingFolders = []protocol.WorkspaceFolder{{
				URI:  string(params.RootURI),
				Name: path.Base(params.RootURI.SpanURI().Filename()),
			}}
		} else {
			// No folders and no root--we are in single file mode.
			// TODO: https://golang.org/issue/34160.
			return nil, errors.New("gopls does not yet support editing a single file. Please open a directory.")
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

	if options.DynamicWatchedFilesSupported {
		registrations = append(registrations, protocol.Registration{
			ID:     "workspace/didChangeWatchedFiles",
			Method: "workspace/didChangeWatchedFiles",
			RegisterOptions: protocol.DidChangeWatchedFilesRegistrationOptions{
				Watchers: []protocol.FileSystemWatcher{{
					GlobPattern: "**/*.go",
					Kind:        float64(protocol.WatchChange + protocol.WatchDelete + protocol.WatchCreate),
				}},
			},
		})
	}

	if len(registrations) > 0 {
		s.client.RegisterCapability(ctx, &protocol.RegistrationParams{
			Registrations: registrations,
		})
	}

	// TODO: this event logging may be unnecessary. The version info is included in the initialize response.
	buf := &bytes.Buffer{}
	debug.PrintVersionInfo(ctx, buf, true, debug.PlainText)
	event.Log(ctx, buf.String())

	s.addFolders(ctx, s.pendingFolders)
	s.pendingFolders = nil

	return nil
}

func (s *Server) addFolders(ctx context.Context, folders []protocol.WorkspaceFolder) {
	originalViews := len(s.session.Views())
	viewErrors := make(map[span.URI]error)

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
			event.Error(ctx, "failed to write environment", err, tag.Directory.Of(view.Folder()))
			continue
		}
		event.Log(ctx, buf.String())

		// Diagnose the newly created view.
		go s.diagnoseDetached(snapshot)
	}
	if len(viewErrors) > 0 {
		errMsg := fmt.Sprintf("Error loading workspace folders (expected %v, got %v)\n", len(folders), len(s.session.Views())-originalViews)
		for uri, err := range viewErrors {
			errMsg += fmt.Sprintf("failed to load view for %s: %v\n", uri, err)
		}
		s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
			Type:    protocol.Error,
			Message: errMsg,
		})
	}
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
		return err
	}
	for _, config := range configs {
		results := source.SetOptions(o, config)
		for _, result := range results {
			if result.Error != nil {
				s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
					Type:    protocol.Error,
					Message: result.Error.Error(),
				})
			}
			switch result.State {
			case source.OptionUnexpected:
				s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
					Type:    protocol.Error,
					Message: fmt.Sprintf("unexpected config %s", result.Name),
				})
			case source.OptionDeprecated:
				msg := fmt.Sprintf("config %s is deprecated", result.Name)
				if result.Replacement != "" {
					msg = fmt.Sprintf("%s, use %s instead", msg, result.Replacement)
				}
				s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
					Type:    protocol.Warning,
					Message: msg,
				})
			}
		}
	}
	return nil
}

// beginFileRequest checks preconditions for a file-oriented request and routes
// it to a snapshot.
// We don't want to return errors for benign conditions like wrong file type,
// so callers should do if !ok { return err } rather than if err != nil.
func (s *Server) beginFileRequest(pURI protocol.DocumentURI, expectKind source.FileKind) (source.Snapshot, source.FileHandle, bool, error) {
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
	fh, err := snapshot.GetFile(uri)
	if err != nil {
		return nil, nil, false, err
	}
	if expectKind != source.UnknownKind && fh.Identity().Kind != expectKind {
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

// ServerExitFunc is used to exit when requested by the client. It is mutable
// for testing purposes.
var ServerExitFunc = os.Exit

func (s *Server) exit(ctx context.Context) error {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	if s.state != serverShutDown {
		ServerExitFunc(1)
	}
	ServerExitFunc(0)
	return nil
}
