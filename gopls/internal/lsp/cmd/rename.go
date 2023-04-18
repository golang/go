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
	"path/filepath"
	"sort"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/diff"
	"golang.org/x/tools/internal/tool"
)

// rename implements the rename verb for gopls.
type rename struct {
	Diff     bool `flag:"d,diff" help:"display diffs instead of rewriting files"`
	Write    bool `flag:"w,write" help:"write result to (source) file instead of stdout"`
	Preserve bool `flag:"preserve" help:"preserve original files"`

	app *Application
}

func (r *rename) Name() string      { return "rename" }
func (r *rename) Parent() string    { return r.app.Name() }
func (r *rename) Usage() string     { return "[rename-flags] <position> <name>" }
func (r *rename) ShortHelp() string { return "rename selected identifier" }
func (r *rename) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Example:

	$ # 1-based location (:line:column or :#position) of the thing to change
	$ gopls rename helper/helper.go:8:6 Foo
	$ gopls rename helper/helper.go:#53 Foo

rename-flags:
`)
	printFlagDefaults(f)
}

// Run renames the specified identifier and either;
// - if -w is specified, updates the file(s) in place;
// - if -d is specified, prints out unified diffs of the changes; or
// - otherwise, prints the new versions to stdout.
func (r *rename) Run(ctx context.Context, args ...string) error {
	if len(args) != 2 {
		return tool.CommandLineErrorf("definition expects 2 arguments (position, new name)")
	}
	conn, err := r.app.connect(ctx, nil)
	if err != nil {
		return err
	}
	defer conn.terminate(ctx)

	from := span.Parse(args[0])
	file, err := conn.openFile(ctx, from.URI())
	if err != nil {
		return err
	}
	loc, err := file.mapper.SpanLocation(from)
	if err != nil {
		return err
	}
	p := protocol.RenameParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
		Position:     loc.Range.Start,
		NewName:      args[1],
	}
	edit, err := conn.Rename(ctx, &p)
	if err != nil {
		return err
	}
	var orderedURIs []string
	edits := map[span.URI][]protocol.TextEdit{}
	for _, c := range edit.DocumentChanges {
		if c.TextDocumentEdit != nil {
			uri := fileURI(c.TextDocumentEdit.TextDocument.URI)
			edits[uri] = append(edits[uri], c.TextDocumentEdit.Edits...)
			orderedURIs = append(orderedURIs, string(uri))
		}
	}
	sort.Strings(orderedURIs)
	changeCount := len(orderedURIs)

	for _, u := range orderedURIs {
		uri := span.URIFromURI(u)
		cmdFile, err := conn.openFile(ctx, uri)
		if err != nil {
			return err
		}
		filename := cmdFile.uri.Filename()

		newContent, renameEdits, err := source.ApplyProtocolEdits(cmdFile.mapper, edits[uri])
		if err != nil {
			return fmt.Errorf("%v: %v", edits, err)
		}

		switch {
		case r.Write:
			fmt.Fprintln(os.Stderr, filename)
			if r.Preserve {
				if err := os.Rename(filename, filename+".orig"); err != nil {
					return fmt.Errorf("%v: %v", edits, err)
				}
			}
			ioutil.WriteFile(filename, newContent, 0644)
		case r.Diff:
			unified, err := diff.ToUnified(filename+".orig", filename, string(cmdFile.mapper.Content), renameEdits)
			if err != nil {
				return err
			}
			fmt.Print(unified)
		default:
			if len(orderedURIs) > 1 {
				fmt.Printf("%s:\n", filepath.Base(filename))
			}
			os.Stdout.Write(newContent)
			if changeCount > 1 { // if this wasn't last change, print newline
				fmt.Println()
			}
			changeCount -= 1
		}
	}
	return nil
}
