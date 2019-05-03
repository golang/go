// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path"
	"strings"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
)

func (s *Server) initialize(ctx context.Context, params *protocol.InitializeParams) (*protocol.InitializeResult, error) {
	s.initializedMu.Lock()
	defer s.initializedMu.Unlock()
	if s.isInitialized {
		return nil, jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidRequest, "server already initialized")
	}
	s.isInitialized = true // mark server as initialized now

	// TODO(iancottrell): Change this default to protocol.Incremental and remove the option
	s.textDocumentSyncKind = protocol.Full
	if opts, ok := params.InitializationOptions.(map[string]interface{}); ok {
		if opt, ok := opts["incrementalSync"].(bool); ok && opt {
			s.textDocumentSyncKind = protocol.Incremental
		}
	}

	s.setClientCapabilities(params.Capabilities)

	// We need a "detached" context so it does not get timeout cancelled.
	// TODO(iancottrell): Do we need to copy any values across?
	viewContext := context.Background()
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
			return nil, fmt.Errorf("single file mode not supported yet")
		}
	}
	for _, folder := range folders {
		uri := span.NewURI(folder.URI)
		folderPath, err := uri.Filename()
		if err != nil {
			return nil, err
		}
		s.views = append(s.views, cache.NewView(viewContext, s.log, folder.Name, uri, &packages.Config{
			Context: ctx,
			Dir:     folderPath,
			Env:     os.Environ(),
			Mode:    packages.LoadImports,
			Fset:    token.NewFileSet(),
			Overlay: make(map[string][]byte),
			ParseFile: func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
				return parser.ParseFile(fset, filename, src, parser.AllErrors|parser.ParseComments)
			},
			Tests: true,
		}))
	}

	return &protocol.InitializeResult{
		Capabilities: protocol.ServerCapabilities{
			CodeActionProvider: true,
			CompletionProvider: &protocol.CompletionOptions{
				TriggerCharacters: []string{"."},
			},
			DefinitionProvider:         true,
			DocumentFormattingProvider: true,
			DocumentSymbolProvider:     true,
			HoverProvider:              true,
			DocumentHighlightProvider:  true,
			DocumentLinkProvider:       &protocol.DocumentLinkOptions{},
			SignatureHelpProvider: &protocol.SignatureHelpOptions{
				TriggerCharacters: []string{"(", ","},
			},
			TextDocumentSync: &protocol.TextDocumentSyncOptions{
				Change:    s.textDocumentSyncKind,
				OpenClose: true,
			},
			TypeDefinitionProvider: true,
		},
	}, nil
}

func (s *Server) setClientCapabilities(caps protocol.ClientCapabilities) {
	// Check if the client supports snippets in completion items.
	s.insertTextFormat = protocol.PlainTextTextFormat
	if caps.TextDocument.Completion.CompletionItem.SnippetSupport {
		s.insertTextFormat = protocol.SnippetTextFormat
	}
	// Check if the client supports configuration messages.
	s.configurationSupported = caps.Workspace.Configuration
	s.dynamicConfigurationSupported = caps.Workspace.DidChangeConfiguration.DynamicRegistration

	// Check which types of content format are supported by this client.
	s.preferredContentFormat = protocol.PlainText
	if len(caps.TextDocument.Hover.ContentFormat) > 0 {
		s.preferredContentFormat = caps.TextDocument.Hover.ContentFormat[0]
	}
}

func (s *Server) initialized(ctx context.Context, params *protocol.InitializedParams) error {
	if s.configurationSupported {
		if s.dynamicConfigurationSupported {
			s.client.RegisterCapability(ctx, &protocol.RegistrationParams{
				Registrations: []protocol.Registration{{
					ID:     "workspace/didChangeConfiguration",
					Method: "workspace/didChangeConfiguration",
				}},
			})
		}
		for _, view := range s.views {
			config, err := s.client.Configuration(ctx, &protocol.ConfigurationParams{
				Items: []protocol.ConfigurationItem{{
					ScopeURI: protocol.NewURI(view.Folder),
					Section:  "gopls",
				}},
			})
			if err != nil {
				return err
			}
			if err := s.processConfig(view, config[0]); err != nil {
				return err
			}
		}
	}
	return nil
}

func (s *Server) processConfig(view *cache.View, config interface{}) error {
	// TODO: We should probably store and process more of the config.
	if config == nil {
		return nil // ignore error if you don't have a config
	}
	c, ok := config.(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid config gopls type %T", config)
	}
	// Get the environment for the go/packages config.
	if env := c["env"]; env != nil {
		menv, ok := env.(map[string]interface{})
		if !ok {
			return fmt.Errorf("invalid config gopls.env type %T", env)
		}
		for k, v := range menv {
			view.Config.Env = applyEnv(view.Config.Env, k, v)
		}
	}
	// Check if placeholders are enabled.
	if usePlaceholders, ok := c["usePlaceholders"].(bool); ok {
		s.usePlaceholders = usePlaceholders
	}
	// Check if enhancedHover is enabled.
	if enhancedHover, ok := c["enhancedHover"].(bool); ok {
		s.enhancedHover = enhancedHover
	}
	return nil
}

func applyEnv(env []string, k string, v interface{}) []string {
	prefix := k + "="
	value := prefix + fmt.Sprint(v)
	for i, s := range env {
		if strings.HasPrefix(s, prefix) {
			env[i] = value
			return env
		}
	}
	return append(env, value)
}

func (s *Server) shutdown(ctx context.Context) error {
	// TODO(rstambler): Cancel contexts here?
	s.initializedMu.Lock()
	defer s.initializedMu.Unlock()
	if !s.isInitialized {
		return jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidRequest, "server not initialized")
	}
	s.isInitialized = false
	return nil
}

func (s *Server) exit(ctx context.Context) error {
	if s.isInitialized {
		os.Exit(1)
	}
	os.Exit(0)
	return nil
}
