// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"golang.org/x/tools/gopls/internal/lsp"
	"golang.org/x/tools/gopls/internal/lsp/cache"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/internal/testenv"
)

// TestCapabilities does some minimal validation of the server's adherence to the LSP.
// The checks in the test are added as changes are made and errors noticed.
func TestCapabilities(t *testing.T) {
	// TODO(bcmills): This test fails on js/wasm, which is not unexpected, but the
	// failure mode is that the DidOpen call below reports "no views in session",
	// which seems a little too cryptic.
	// Is there some missing error reporting somewhere?
	testenv.NeedsTool(t, "go")

	tmpDir, err := os.MkdirTemp("", "fake")
	if err != nil {
		t.Fatal(err)
	}
	tmpFile := filepath.Join(tmpDir, "fake.go")
	if err := os.WriteFile(tmpFile, []byte(""), 0775); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(tmpDir, "go.mod"), []byte("module fake\n\ngo 1.12\n"), 0775); err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	app := New("gopls-test", tmpDir, os.Environ(), nil)

	params := &protocol.ParamInitialize{}
	params.RootURI = protocol.URIFromPath(app.wd)
	params.Capabilities.Workspace.Configuration = true

	// Send an initialize request to the server.
	ctx := context.Background()
	client := newClient(app, nil)
	options := source.DefaultOptions(app.options)
	server := lsp.NewServer(cache.NewSession(ctx, cache.New(nil)), client, options)
	result, err := server.Initialize(ctx, params)
	if err != nil {
		t.Fatal(err)
	}
	// Validate initialization result.
	if err := validateCapabilities(result); err != nil {
		t.Error(err)
	}
	// Complete initialization of server.
	if err := server.Initialized(ctx, &protocol.InitializedParams{}); err != nil {
		t.Fatal(err)
	}

	c := newConnection(server, client)
	defer c.terminate(ctx)

	// Open the file on the server side.
	uri := protocol.URIFromPath(tmpFile)
	if err := c.Server.DidOpen(ctx, &protocol.DidOpenTextDocumentParams{
		TextDocument: protocol.TextDocumentItem{
			URI:        uri,
			LanguageID: "go",
			Version:    1,
			Text:       `package main; func main() {};`,
		},
	}); err != nil {
		t.Fatal(err)
	}

	// If we are sending a full text change, the change.Range must be nil.
	// It is not enough for the Change to be empty, as that is ambiguous.
	if err := c.Server.DidChange(ctx, &protocol.DidChangeTextDocumentParams{
		TextDocument: protocol.VersionedTextDocumentIdentifier{
			TextDocumentIdentifier: protocol.TextDocumentIdentifier{
				URI: uri,
			},
			Version: 2,
		},
		ContentChanges: []protocol.TextDocumentContentChangeEvent{
			{
				Range: nil,
				Text:  `package main; func main() { fmt.Println("") }`,
			},
		},
	}); err != nil {
		t.Fatal(err)
	}

	// Send a code action request to validate expected types.
	actions, err := c.Server.CodeAction(ctx, &protocol.CodeActionParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: uri,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, action := range actions {
		// Validate that an empty command is sent along with import organization responses.
		if action.Kind == protocol.SourceOrganizeImports && action.Command != nil {
			t.Errorf("unexpected command for import organization")
		}
	}

	if err := c.Server.DidSave(ctx, &protocol.DidSaveTextDocumentParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: uri,
		},
		// LSP specifies that a file can be saved with optional text, so this field must be nil.
		Text: nil,
	}); err != nil {
		t.Fatal(err)
	}

	// Send a completion request to validate expected types.
	list, err := c.Server.Completion(ctx, &protocol.CompletionParams{
		TextDocumentPositionParams: protocol.TextDocumentPositionParams{
			TextDocument: protocol.TextDocumentIdentifier{
				URI: uri,
			},
			Position: protocol.Position{
				Line:      0,
				Character: 28,
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, item := range list.Items {
		// All other completion items should have nil commands.
		// An empty command will be treated as a command with the name '' by VS Code.
		// This causes VS Code to report errors to users about invalid commands.
		if item.Command != nil {
			t.Errorf("unexpected command for completion item")
		}
		// The item's TextEdit must be a pointer, as VS Code considers TextEdits
		// that don't contain the cursor position to be invalid.
		var textEdit interface{} = item.TextEdit
		if _, ok := textEdit.(*protocol.TextEdit); !ok {
			t.Errorf("textEdit is not a *protocol.TextEdit, instead it is %T", textEdit)
		}
	}
	if err := c.Server.Shutdown(ctx); err != nil {
		t.Fatal(err)
	}
	if err := c.Server.Exit(ctx); err != nil {
		t.Fatal(err)
	}
}

func validateCapabilities(result *protocol.InitializeResult) error {
	// If the client sends "false" for RenameProvider.PrepareSupport,
	// the server must respond with a boolean.
	if v, ok := result.Capabilities.RenameProvider.(bool); !ok {
		return fmt.Errorf("RenameProvider must be a boolean if PrepareSupport is false (got %T)", v)
	}
	// The same goes for CodeActionKind.ValueSet.
	if v, ok := result.Capabilities.CodeActionProvider.(bool); !ok {
		return fmt.Errorf("CodeActionSupport must be a boolean if CodeActionKind.ValueSet has length 0 (got %T)", v)
	}
	return nil
}
