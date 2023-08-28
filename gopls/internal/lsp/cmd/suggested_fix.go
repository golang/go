// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"flag"
	"fmt"
	"io/ioutil"
	"os"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/diff"
	"golang.org/x/tools/internal/tool"
)

// suggestedFix implements the fix verb for gopls.
type suggestedFix struct {
	Diff  bool `flag:"d,diff" help:"display diffs instead of rewriting files"`
	Write bool `flag:"w,write" help:"write result to (source) file instead of stdout"`
	All   bool `flag:"a,all" help:"apply all fixes, not just preferred fixes"`

	app *Application
}

func (s *suggestedFix) Name() string      { return "fix" }
func (s *suggestedFix) Parent() string    { return s.app.Name() }
func (s *suggestedFix) Usage() string     { return "[fix-flags] <filename>" }
func (s *suggestedFix) ShortHelp() string { return "apply suggested fixes" }
func (s *suggestedFix) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprintf(f.Output(), `
Example: apply suggested fixes for this file
	$ gopls fix -w internal/lsp/cmd/check.go

fix-flags:
`)
	printFlagDefaults(f)
}

// Run performs diagnostic checks on the file specified and either;
// - if -w is specified, updates the file in place;
// - if -d is specified, prints out unified diffs of the changes; or
// - otherwise, prints the new versions to stdout.
func (s *suggestedFix) Run(ctx context.Context, args ...string) error {
	if len(args) < 1 {
		return tool.CommandLineErrorf("fix expects at least 1 argument")
	}
	conn, err := s.app.connect(ctx, nil)
	if err != nil {
		return err
	}
	defer conn.terminate(ctx)

	from := span.Parse(args[0])
	uri := from.URI()
	file, err := conn.openFile(ctx, uri)
	if err != nil {
		return err
	}
	rng, err := file.mapper.SpanRange(from)
	if err != nil {
		return err
	}

	// Get diagnostics.
	if err := conn.diagnoseFiles(ctx, []span.URI{uri}); err != nil {
		return err
	}
	diagnostics := []protocol.Diagnostic{} // LSP wants non-nil slice
	conn.client.filesMu.Lock()
	diagnostics = append(diagnostics, file.diagnostics...)
	conn.client.filesMu.Unlock()

	// Request code actions
	codeActionKinds := []protocol.CodeActionKind{protocol.QuickFix}
	if len(args) > 1 {
		codeActionKinds = []protocol.CodeActionKind{}
		for _, k := range args[1:] {
			codeActionKinds = append(codeActionKinds, protocol.CodeActionKind(k))
		}
	}
	p := protocol.CodeActionParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(uri),
		},
		Context: protocol.CodeActionContext{
			Only:        codeActionKinds,
			Diagnostics: diagnostics,
		},
		Range: rng,
	}
	actions, err := conn.CodeAction(ctx, &p)
	if err != nil {
		return fmt.Errorf("%v: %v", from, err)
	}

	// Gather edits from matching code actions.
	var edits []protocol.TextEdit
	for _, a := range actions {
		// Without -all, apply only "preferred" fixes.
		if !a.IsPreferred && !s.All {
			continue
		}

		// Execute any command.
		// This may cause the server to make
		// an ApplyEdit downcall to the client.
		if a.Command != nil {
			if _, err := conn.ExecuteCommand(ctx, &protocol.ExecuteCommandParams{
				Command:   a.Command.Command,
				Arguments: a.Command.Arguments,
			}); err != nil {
				return err
			}
			// The specification says that commands should
			// be executed _after_ edits are applied, not
			// instead of them, but we don't want to
			// duplicate edits.
			continue
		}

		// Partially apply CodeAction.Edit, a WorkspaceEdit.
		// (See also conn.Client.applyWorkspaceEdit(a.Edit)).
		if !from.HasPosition() {
			for _, c := range a.Edit.DocumentChanges {
				if c.TextDocumentEdit != nil {
					if fileURI(c.TextDocumentEdit.TextDocument.URI) == uri {
						edits = append(edits, c.TextDocumentEdit.Edits...)
					}
				}
			}
			continue
		}

		// The provided span has a position (not just offsets).
		// Find the code action that has the same range as it.
		for _, diag := range a.Diagnostics {
			if diag.Range.Start == rng.Start {
				for _, c := range a.Edit.DocumentChanges {
					if c.TextDocumentEdit != nil {
						if fileURI(c.TextDocumentEdit.TextDocument.URI) == uri {
							edits = append(edits, c.TextDocumentEdit.Edits...)
						}
					}
				}
				break
			}
		}

		// If suggested fix is not a diagnostic, still must collect edits.
		if len(a.Diagnostics) == 0 {
			for _, c := range a.Edit.DocumentChanges {
				if c.TextDocumentEdit != nil {
					if fileURI(c.TextDocumentEdit.TextDocument.URI) == uri {
						edits = append(edits, c.TextDocumentEdit.Edits...)
					}
				}
			}
		}
	}

	newContent, sedits, err := source.ApplyProtocolEdits(file.mapper, edits)
	if err != nil {
		return fmt.Errorf("%v: %v", edits, err)
	}

	filename := file.uri.Filename()
	switch {
	case s.Write:
		if len(edits) > 0 {
			ioutil.WriteFile(filename, newContent, 0644)
		}
	case s.Diff:
		diffs, err := diff.ToUnified(filename+".orig", filename, string(file.mapper.Content), sedits)
		if err != nil {
			return err
		}
		fmt.Print(diffs)
	default:
		os.Stdout.Write(newContent)
	}
	return nil
}
