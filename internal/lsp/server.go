// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"net"
	"os"
	"sync"
	"unicode/utf8"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/jsonrpc2"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

// RunServer starts an LSP server on the supplied stream, and waits until the
// stream is closed.
func RunServer(ctx context.Context, stream jsonrpc2.Stream, opts ...interface{}) error {
	s := &server{}
	conn, client := protocol.RunServer(ctx, stream, s, opts...)
	s.client = client
	return conn.Wait(ctx)
}

// RunServerOnPort starts an LSP server on the given port and does not exit.
// This function exists for debugging purposes.
func RunServerOnPort(ctx context.Context, port int, opts ...interface{}) error {
	return RunServerOnAddress(ctx, fmt.Sprintf(":%v", port))
}

// RunServerOnPort starts an LSP server on the given port and does not exit.
// This function exists for debugging purposes.
func RunServerOnAddress(ctx context.Context, addr string, opts ...interface{}) error {
	s := &server{}
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	for {
		conn, err := ln.Accept()
		if err != nil {
			return err
		}
		stream := jsonrpc2.NewHeaderStream(conn, conn)
		go func() {
			conn, client := protocol.RunServer(ctx, stream, s, opts...)
			s.client = client
			conn.Wait(ctx)
		}()
	}
}

type server struct {
	client protocol.Client

	initializedMu sync.Mutex
	initialized   bool // set once the server has received "initialize" request

	signatureHelpEnabled bool
	snippetsSupported    bool

	viewMu sync.Mutex
	view   source.View
}

func (s *server) Initialize(ctx context.Context, params *protocol.InitializeParams) (*protocol.InitializeResult, error) {
	s.initializedMu.Lock()
	defer s.initializedMu.Unlock()
	if s.initialized {
		return nil, jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidRequest, "server already initialized")
	}
	s.initialized = true // mark server as initialized now

	// Check if the client supports snippets in completion items.
	s.snippetsSupported = params.Capabilities.TextDocument.Completion.CompletionItem.SnippetSupport
	s.signatureHelpEnabled = true

	var rootURI protocol.DocumentURI
	if params.RootURI != nil {
		rootURI = *params.RootURI
	}
	sourceURI, err := fromProtocolURI(rootURI)
	if err != nil {
		return nil, err
	}
	rootPath, err := sourceURI.Filename()
	if err != nil {
		return nil, err
	}

	s.view = cache.NewView(&packages.Config{
		Context: ctx,
		Dir:     rootPath,
		Mode:    packages.LoadImports,
		Fset:    token.NewFileSet(),
		Overlay: make(map[string][]byte),
		ParseFile: func(fset *token.FileSet, filename string, src []byte) (*ast.File, error) {
			return parser.ParseFile(fset, filename, src, parser.AllErrors|parser.ParseComments)
		},
		Tests: true,
	})

	return &protocol.InitializeResult{
		Capabilities: protocol.ServerCapabilities{
			CodeActionProvider: true,
			CompletionProvider: protocol.CompletionOptions{
				TriggerCharacters: []string{"."},
			},
			DefinitionProvider:              true,
			DocumentFormattingProvider:      true,
			DocumentRangeFormattingProvider: true,
			HoverProvider:                   true,
			SignatureHelpProvider: protocol.SignatureHelpOptions{
				TriggerCharacters: []string{"(", ","},
			},
			TextDocumentSync: protocol.TextDocumentSyncOptions{
				Change:    float64(protocol.Incremental),
				OpenClose: true,
			},
			TypeDefinitionProvider: true,
		},
	}, nil
}

func (s *server) Initialized(context.Context, *protocol.InitializedParams) error {
	return nil // ignore
}

func (s *server) Shutdown(context.Context) error {
	s.initializedMu.Lock()
	defer s.initializedMu.Unlock()
	if !s.initialized {
		return jsonrpc2.NewErrorf(jsonrpc2.CodeInvalidRequest, "server not initialized")
	}
	s.initialized = false
	return nil
}

func (s *server) Exit(ctx context.Context) error {
	if s.initialized {
		os.Exit(1)
	}
	os.Exit(0)
	return nil
}

func (s *server) DidChangeWorkspaceFolders(context.Context, *protocol.DidChangeWorkspaceFoldersParams) error {
	return notImplemented("DidChangeWorkspaceFolders")
}

func (s *server) DidChangeConfiguration(context.Context, *protocol.DidChangeConfigurationParams) error {
	return notImplemented("DidChangeConfiguration")
}

func (s *server) DidChangeWatchedFiles(context.Context, *protocol.DidChangeWatchedFilesParams) error {
	return notImplemented("DidChangeWatchedFiles")
}

func (s *server) Symbols(context.Context, *protocol.WorkspaceSymbolParams) ([]protocol.SymbolInformation, error) {
	return nil, notImplemented("Symbols")
}

func (s *server) ExecuteCommand(context.Context, *protocol.ExecuteCommandParams) (interface{}, error) {
	return nil, notImplemented("ExecuteCommand")
}

func (s *server) DidOpen(ctx context.Context, params *protocol.DidOpenTextDocumentParams) error {
	s.cacheAndDiagnose(ctx, params.TextDocument.URI, params.TextDocument.Text)
	return nil
}

func bytesOffset(content []byte, pos protocol.Position) int {
	var line, char, offset int

	for len(content) > 0 {
		if line == int(pos.Line) && char == int(pos.Character) {
			return offset
		}
		r, size := utf8.DecodeRune(content)
		char++
		// The offsets are based on a UTF-16 string representation.
		// So the rune should be checked twice for two code units in UTF-16.
		if r >= 0x10000 {
			if line == int(pos.Line) && char == int(pos.Character) {
				return offset
			}
			char++
		}
		offset += size
		content = content[size:]
		if r == '\n' {
			line++
			char = 0
		}
	}
	return -1
}

func (s *server) applyChanges(ctx context.Context, params *protocol.DidChangeTextDocumentParams) (string, error) {
	if len(params.ContentChanges) == 1 && params.ContentChanges[0].Range == nil {
		// If range is empty, we expect the full content of file, i.e. a single change with no range.
		change := params.ContentChanges[0]
		if change.RangeLength != 0 {
			return "", jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "unexpected change range provided")
		}
		return change.Text, nil
	}

	sourceURI, err := fromProtocolURI(params.TextDocument.URI)
	if err != nil {
		return "", err
	}

	file, err := s.view.GetFile(ctx, sourceURI)
	if err != nil {
		return "", jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "file not found")
	}

	content := file.GetContent()
	for _, change := range params.ContentChanges {
		start := bytesOffset(content, change.Range.Start)
		if start == -1 {
			return "", jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "invalid range for content change")
		}
		end := bytesOffset(content, change.Range.End)
		if end == -1 {
			return "", jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "invalid range for content change")
		}
		var buf bytes.Buffer
		buf.Write(content[:start])
		buf.WriteString(change.Text)
		buf.Write(content[end:])
		content = buf.Bytes()
	}
	return string(content), nil
}

func (s *server) DidChange(ctx context.Context, params *protocol.DidChangeTextDocumentParams) error {
	if len(params.ContentChanges) < 1 {
		return jsonrpc2.NewErrorf(jsonrpc2.CodeInternalError, "no content changes provided")
	}

	text, err := s.applyChanges(ctx, params)
	if err != nil {
		return err
	}
	s.cacheAndDiagnose(ctx, params.TextDocument.URI, text)
	return nil
}

func (s *server) WillSave(context.Context, *protocol.WillSaveTextDocumentParams) error {
	return notImplemented("WillSave")
}

func (s *server) WillSaveWaitUntil(context.Context, *protocol.WillSaveTextDocumentParams) ([]protocol.TextEdit, error) {
	return nil, notImplemented("WillSaveWaitUntil")
}

func (s *server) DidSave(context.Context, *protocol.DidSaveTextDocumentParams) error {
	return nil // ignore
}

func (s *server) DidClose(ctx context.Context, params *protocol.DidCloseTextDocumentParams) error {
	sourceURI, err := fromProtocolURI(params.TextDocument.URI)
	if err != nil {
		return err
	}
	s.setContent(ctx, sourceURI, nil)
	return nil
}

func (s *server) Completion(ctx context.Context, params *protocol.CompletionParams) (*protocol.CompletionList, error) {
	sourceURI, err := fromProtocolURI(params.TextDocument.URI)
	if err != nil {
		return nil, err
	}
	f, err := s.view.GetFile(ctx, sourceURI)
	if err != nil {
		return nil, err
	}
	tok := f.GetToken()
	pos := fromProtocolPosition(tok, params.Position)
	items, prefix, err := source.Completion(ctx, f, pos)
	if err != nil {
		return nil, err
	}
	return &protocol.CompletionList{
		IsIncomplete: false,
		Items:        toProtocolCompletionItems(items, prefix, params.Position, s.snippetsSupported, s.signatureHelpEnabled),
	}, nil
}

func (s *server) CompletionResolve(context.Context, *protocol.CompletionItem) (*protocol.CompletionItem, error) {
	return nil, notImplemented("CompletionResolve")
}

func (s *server) Hover(ctx context.Context, params *protocol.TextDocumentPositionParams) (*protocol.Hover, error) {
	sourceURI, err := fromProtocolURI(params.TextDocument.URI)
	if err != nil {
		return nil, err
	}
	f, err := s.view.GetFile(ctx, sourceURI)
	if err != nil {
		return nil, err
	}
	tok := f.GetToken()
	pos := fromProtocolPosition(tok, params.Position)
	ident, err := source.Identifier(ctx, s.view, f, pos)
	if err != nil {
		return nil, err
	}
	content, err := ident.Hover(nil)
	if err != nil {
		return nil, err
	}
	markdown := "```go\n" + content + "\n```"
	return &protocol.Hover{
		Contents: protocol.MarkupContent{
			Kind:  protocol.Markdown,
			Value: markdown,
		},
		Range: toProtocolRange(tok, ident.Range),
	}, nil
}

func (s *server) SignatureHelp(ctx context.Context, params *protocol.TextDocumentPositionParams) (*protocol.SignatureHelp, error) {
	sourceURI, err := fromProtocolURI(params.TextDocument.URI)
	if err != nil {
		return nil, err
	}
	f, err := s.view.GetFile(ctx, sourceURI)
	if err != nil {
		return nil, err
	}
	tok := f.GetToken()
	pos := fromProtocolPosition(tok, params.Position)
	info, err := source.SignatureHelp(ctx, f, pos)
	if err != nil {
		return nil, err
	}
	return toProtocolSignatureHelp(info), nil
}

func (s *server) Definition(ctx context.Context, params *protocol.TextDocumentPositionParams) ([]protocol.Location, error) {
	sourceURI, err := fromProtocolURI(params.TextDocument.URI)
	if err != nil {
		return nil, err
	}
	f, err := s.view.GetFile(ctx, sourceURI)
	if err != nil {
		return nil, err
	}
	tok := f.GetToken()
	pos := fromProtocolPosition(tok, params.Position)
	ident, err := source.Identifier(ctx, s.view, f, pos)
	if err != nil {
		return nil, err
	}
	return []protocol.Location{toProtocolLocation(s.view.FileSet(), ident.Declaration.Range)}, nil
}

func (s *server) TypeDefinition(ctx context.Context, params *protocol.TextDocumentPositionParams) ([]protocol.Location, error) {
	sourceURI, err := fromProtocolURI(params.TextDocument.URI)
	if err != nil {
		return nil, err
	}
	f, err := s.view.GetFile(ctx, sourceURI)
	if err != nil {
		return nil, err
	}
	tok := f.GetToken()
	pos := fromProtocolPosition(tok, params.Position)
	ident, err := source.Identifier(ctx, s.view, f, pos)
	if err != nil {
		return nil, err
	}
	return []protocol.Location{toProtocolLocation(s.view.FileSet(), ident.Type.Range)}, nil
}

func (s *server) Implementation(context.Context, *protocol.TextDocumentPositionParams) ([]protocol.Location, error) {
	return nil, notImplemented("Implementation")
}

func (s *server) References(context.Context, *protocol.ReferenceParams) ([]protocol.Location, error) {
	return nil, notImplemented("References")
}

func (s *server) DocumentHighlight(context.Context, *protocol.TextDocumentPositionParams) ([]protocol.DocumentHighlight, error) {
	return nil, notImplemented("DocumentHighlight")
}

func (s *server) DocumentSymbol(context.Context, *protocol.DocumentSymbolParams) ([]protocol.DocumentSymbol, error) {
	return nil, notImplemented("DocumentSymbol")
}

func (s *server) CodeAction(ctx context.Context, params *protocol.CodeActionParams) ([]protocol.CodeAction, error) {
	edits, err := organizeImports(ctx, s.view, params.TextDocument.URI)
	if err != nil {
		return nil, err
	}
	return []protocol.CodeAction{
		{
			Title: "Organize Imports",
			Kind:  protocol.SourceOrganizeImports,
			Edit: protocol.WorkspaceEdit{
				Changes: map[protocol.DocumentURI][]protocol.TextEdit{
					params.TextDocument.URI: edits,
				},
			},
		},
	}, nil
}

func (s *server) CodeLens(context.Context, *protocol.CodeLensParams) ([]protocol.CodeLens, error) {
	return nil, nil // ignore
}

func (s *server) CodeLensResolve(context.Context, *protocol.CodeLens) (*protocol.CodeLens, error) {
	return nil, notImplemented("CodeLensResolve")
}

func (s *server) DocumentLink(context.Context, *protocol.DocumentLinkParams) ([]protocol.DocumentLink, error) {
	return nil, nil // ignore
}

func (s *server) DocumentLinkResolve(context.Context, *protocol.DocumentLink) (*protocol.DocumentLink, error) {
	return nil, notImplemented("DocumentLinkResolve")
}

func (s *server) DocumentColor(context.Context, *protocol.DocumentColorParams) ([]protocol.ColorInformation, error) {
	return nil, notImplemented("DocumentColor")
}

func (s *server) ColorPresentation(context.Context, *protocol.ColorPresentationParams) ([]protocol.ColorPresentation, error) {
	return nil, notImplemented("ColorPresentation")
}

func (s *server) Formatting(ctx context.Context, params *protocol.DocumentFormattingParams) ([]protocol.TextEdit, error) {
	return formatRange(ctx, s.view, params.TextDocument.URI, nil)
}

func (s *server) RangeFormatting(ctx context.Context, params *protocol.DocumentRangeFormattingParams) ([]protocol.TextEdit, error) {
	return formatRange(ctx, s.view, params.TextDocument.URI, &params.Range)
}

func (s *server) OnTypeFormatting(context.Context, *protocol.DocumentOnTypeFormattingParams) ([]protocol.TextEdit, error) {
	return nil, notImplemented("OnTypeFormatting")
}

func (s *server) Rename(context.Context, *protocol.RenameParams) ([]protocol.WorkspaceEdit, error) {
	return nil, notImplemented("Rename")
}

func (s *server) FoldingRanges(context.Context, *protocol.FoldingRangeRequestParam) ([]protocol.FoldingRange, error) {
	return nil, notImplemented("FoldingRanges")
}

func notImplemented(method string) *jsonrpc2.Error {
	return jsonrpc2.NewErrorf(jsonrpc2.CodeMethodNotFound, "method %q not yet implemented", method)
}
