// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"path"

	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/debug"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/tag"
	errors "golang.org/x/xerrors"
)

func (s *Server) initialize(ctx context.Context, params *protocol.InitializeParams) (*protocol.InitializeResult, error) {
	s.stateMu.Lock()
	state := s.state
	s.stateMu.Unlock()
	if state >= serverInitializing {
		return nil, jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidRequest, "server already initialized")
	}
	s.stateMu.Lock()
	s.state = serverInitializing
	s.stateMu.Unlock()

	options := s.session.Options()
	defer func() { s.session.SetOptions(options) }()

	// TODO: Remove the option once we are certain there are no issues here.
	options.TextDocumentSyncKind = protocol.Incremental
	if opts, ok := params.InitializationOptions.(map[string]interface{}); ok {
		if opt, ok := opts["noIncrementalSync"].(bool); ok && opt {
			options.TextDocumentSyncKind = protocol.Full
		}

		// Check if user has enabled watching for file changes.
		setBool(&options.WatchFileChanges, opts, "watchFileChanges")
	}

	// Default to using synopsis as a default for hover information.
	options.HoverKind = source.SynopsisDocumentation

	options.SupportedCodeActions = map[source.FileKind]map[protocol.CodeActionKind]bool{
		source.Go: {
			protocol.SourceOrganizeImports: true,
			protocol.QuickFix:              true,
		},
		source.Mod: {},
		source.Sum: {},
	}

	s.setClientCapabilities(&options, params.Capabilities)

	folders := params.WorkspaceFolders
	if len(folders) == 0 {
		if params.RootURI != "" {
			folders = []protocol.WorkspaceFolder{{
				URI:  params.RootURI,
				Name: path.Base(params.RootURI),
			}}
		} else {
			// no folders and no root, single file mode
			//TODO(iancottrell): not sure how to do single file mode yet
			//issue: golang.org/issue/31168
			return nil, errors.Errorf("single file mode not supported yet")
		}
	}

	for _, folder := range folders {
		if err := s.addView(ctx, folder.Name, span.NewURI(folder.URI)); err != nil {
			return nil, err
		}
	}

	var codeActionProvider interface{}
	if len(params.Capabilities.TextDocument.CodeAction.CodeActionLiteralSupport.CodeActionKind.ValueSet) > 0 {
		// If the client has specified CodeActionLiteralSupport,
		// send the code actions we support.
		//
		// Using CodeActionOptions is only valid if codeActionLiteralSupport is set.
		codeActionProvider = &protocol.CodeActionOptions{
			CodeActionKinds: s.getSupportedCodeActions(),
		}
	} else {
		codeActionProvider = true
	}
	var renameOpts interface{}
	if params.Capabilities.TextDocument.Rename.PrepareSupport {
		renameOpts = &protocol.RenameOptions{
			PrepareProvider: true,
		}
	} else {
		renameOpts = true
	}
	return &protocol.InitializeResult{
		Capabilities: protocol.ServerCapabilities{
			CodeActionProvider: codeActionProvider,
			CompletionProvider: &protocol.CompletionOptions{
				TriggerCharacters: []string{"."},
			},
			DefinitionProvider:         true,
			DocumentFormattingProvider: true,
			DocumentSymbolProvider:     true,
			FoldingRangeProvider:       true,
			HoverProvider:              true,
			DocumentHighlightProvider:  true,
			DocumentLinkProvider:       &protocol.DocumentLinkOptions{},
			ReferencesProvider:         true,
			RenameProvider:             renameOpts,
			SignatureHelpProvider: &protocol.SignatureHelpOptions{
				TriggerCharacters: []string{"(", ","},
			},
			TextDocumentSync: &protocol.TextDocumentSyncOptions{
				Change:    options.TextDocumentSyncKind,
				OpenClose: true,
				Save: &protocol.SaveOptions{
					IncludeText: false,
				},
			},
			TypeDefinitionProvider: true,
			Workspace: &struct {
				WorkspaceFolders *struct {
					Supported           bool   "json:\"supported,omitempty\""
					ChangeNotifications string "json:\"changeNotifications,omitempty\""
				} "json:\"workspaceFolders,omitempty\""
			}{
				WorkspaceFolders: &struct {
					Supported           bool   "json:\"supported,omitempty\""
					ChangeNotifications string "json:\"changeNotifications,omitempty\""
				}{
					Supported:           true,
					ChangeNotifications: "workspace/didChangeWorkspaceFolders",
				},
			},
		},
	}, nil
}

func (s *Server) setClientCapabilities(o *source.SessionOptions, caps protocol.ClientCapabilities) {
	// Check if the client supports snippets in completion items.
	o.InsertTextFormat = protocol.PlainTextTextFormat
	if caps.TextDocument.Completion.CompletionItem.SnippetSupport {
		o.InsertTextFormat = protocol.SnippetTextFormat
	}
	// Check if the client supports configuration messages.
	o.ConfigurationSupported = caps.Workspace.Configuration
	o.DynamicConfigurationSupported = caps.Workspace.DidChangeConfiguration.DynamicRegistration
	o.DynamicWatchedFilesSupported = caps.Workspace.DidChangeWatchedFiles.DynamicRegistration

	// Check which types of content format are supported by this client.
	o.PreferredContentFormat = protocol.PlainText
	if len(caps.TextDocument.Hover.ContentFormat) > 0 {
		o.PreferredContentFormat = caps.TextDocument.Hover.ContentFormat[0]
	}
	// Check if the client supports only line folding.
	o.LineFoldingOnly = caps.TextDocument.FoldingRange.LineFoldingOnly
}

func (s *Server) initialized(ctx context.Context, params *protocol.InitializedParams) error {
	s.stateMu.Lock()
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

	if options.WatchFileChanges && options.DynamicWatchedFilesSupported {
		registrations = append(registrations, protocol.Registration{
			ID:     "workspace/didChangeWatchedFiles",
			Method: "workspace/didChangeWatchedFiles",
			RegisterOptions: protocol.DidChangeWatchedFilesRegistrationOptions{
				Watchers: []protocol.FileSystemWatcher{{
					GlobPattern: "**/*.go",
					Kind:        float64(protocol.WatchChange),
				}},
			},
		})
	}

	if len(registrations) > 0 {
		s.client.RegisterCapability(ctx, &protocol.RegistrationParams{
			Registrations: registrations,
		})
	}

	if options.ConfigurationSupported {
		for _, view := range s.session.Views() {
			if err := s.fetchConfig(ctx, view, &options); err != nil {
				return err
			}
		}
	}
	buf := &bytes.Buffer{}
	debug.PrintVersionInfo(buf, true, debug.PlainText)
	log.Print(ctx, buf.String())
	return nil
}

func (s *Server) fetchConfig(ctx context.Context, view source.View, options *source.SessionOptions) error {
	configs, err := s.client.Configuration(ctx, &protocol.ConfigurationParams{
		Items: []protocol.ConfigurationItem{{
			ScopeURI: protocol.NewURI(view.Folder()),
			Section:  "gopls",
		}, {
			ScopeURI: protocol.NewURI(view.Folder()),
			Section:  view.Name(),
		},
		},
	})
	if err != nil {
		return err
	}
	for _, config := range configs {
		if err := s.processConfig(ctx, view, options, config); err != nil {
			return err
		}
	}
	return nil
}

func (s *Server) processConfig(ctx context.Context, view source.View, options *source.SessionOptions, config interface{}) error {
	// TODO: We should probably store and process more of the config.
	if config == nil {
		return nil // ignore error if you don't have a config
	}

	c, ok := config.(map[string]interface{})
	if !ok {
		return errors.Errorf("invalid config gopls type %T", config)
	}

	// Get the environment for the go/packages config.
	if env := c["env"]; env != nil {
		menv, ok := env.(map[string]interface{})
		if !ok {
			return errors.Errorf("invalid config gopls.env type %T", env)
		}
		env := view.Env()
		for k, v := range menv {
			env = append(env, fmt.Sprintf("%s=%s", k, v))
		}
		view.SetEnv(env)
	}

	// Get the build flags for the go/packages config.
	if buildFlags := c["buildFlags"]; buildFlags != nil {
		iflags, ok := buildFlags.([]interface{})
		if !ok {
			return errors.Errorf("invalid config gopls.buildFlags type %T", buildFlags)
		}
		flags := make([]string, 0, len(iflags))
		for _, flag := range iflags {
			flags = append(flags, fmt.Sprintf("%s", flag))
		}
		view.SetBuildFlags(flags)
	}

	// Check if the user wants documentation in completion items.
	// This defaults to true.
	options.Completion.Documentation = true
	setBool(&options.Completion.Documentation, c, "wantCompletionDocumentation")
	setBool(&options.UsePlaceholders, c, "usePlaceholders")

	// Set the hover kind.
	if hoverKind, ok := c["hoverKind"].(string); ok {
		switch hoverKind {
		case "NoDocumentation":
			options.HoverKind = source.NoDocumentation
		case "SingleLine":
			options.HoverKind = source.SingleLine
		case "SynopsisDocumentation":
			options.HoverKind = source.SynopsisDocumentation
		case "FullDocumentation":
			options.HoverKind = source.FullDocumentation
		case "Structured":
			options.HoverKind = source.Structured
		default:
			log.Error(ctx, "unsupported hover kind", nil, tag.Of("HoverKind", hoverKind))
			// The default value is already be set to synopsis.
		}
	}

	// Check if the user wants to see suggested fixes from go/analysis.
	setBool(&options.SuggestedFixes, c, "wantSuggestedFixes")

	// Check if the user has explicitly disabled any analyses.
	if disabledAnalyses, ok := c["experimentalDisabledAnalyses"].([]interface{}); ok {
		options.DisabledAnalyses = make(map[string]struct{})
		for _, a := range disabledAnalyses {
			if a, ok := a.(string); ok {
				options.DisabledAnalyses[a] = struct{}{}
			}
		}
	}

	setNotBool(&options.Completion.Deep, c, "disableDeepCompletion")
	setNotBool(&options.Completion.FuzzyMatching, c, "disableFuzzyMatching")

	// Check if want unimported package completions.
	setBool(&options.Completion.Unimported, c, "wantUnimportedCompletions")
	return nil
}

func (s *Server) shutdown(ctx context.Context) error {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	if s.state < serverInitialized {
		return jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidRequest, "server not initialized")
	}
	// drop all the active views
	s.session.Shutdown(ctx)
	s.state = serverShutDown
	return nil
}

func (s *Server) exit(ctx context.Context) error {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	if s.state != serverShutDown {
		os.Exit(1)
	}
	os.Exit(0)
	return nil
}

func setBool(b *bool, m map[string]interface{}, name string) {
	if v, ok := m[name].(bool); ok {
		*b = v
	}
}

func setNotBool(b *bool, m map[string]interface{}, name string) {
	if v, ok := m[name].(bool); ok {
		*b = !v
	}
}
