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

	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/tool"
	errors "golang.org/x/xerrors"
)

// rename implements the rename verb for gopls.
type rename struct {
	Diff  bool `flag:"d" help:"display diffs instead of rewriting files"`
	Write bool `flag:"w" help:"write result to (source) file instead of stdout"`

	app *Application
}

func (r *rename) Name() string      { return "rename" }
func (r *rename) Usage() string     { return "<position>" }
func (r *rename) ShortHelp() string { return "rename selected identifier" }
func (r *rename) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Example:

  $ # 1-based location (:line:column or :#position) of the thing to change
  $ gopls rename helper/helper.go:8:6
  $ gopls rename helper/helper.go:#53

	gopls rename flags are:
`)
	f.PrintDefaults()
}

// Run renames the specified identifier and either;
// - if -w is specified, updates the file(s) in place;
// - if -d is specified, prints out unified diffs of the changes; or
// - otherwise, prints the new versions to stdout.
func (r *rename) Run(ctx context.Context, args ...string) error {
	if len(args) != 2 {
		return tool.CommandLineErrorf("definition expects 2 arguments (position, new name)")
	}
	conn, err := r.app.connect(ctx)
	if err != nil {
		return err
	}
	defer conn.terminate(ctx)

	from := span.Parse(args[0])
	file := conn.AddFile(ctx, from.URI())
	if file.err != nil {
		return file.err
	}

	loc, err := file.mapper.Location(from)
	if err != nil {
		return err
	}

	p := protocol.RenameParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
		Position:     loc.Range.Start,
		NewName:      args[1],
	}
	we, err := conn.Rename(ctx, &p)
	if err != nil {
		return err
	}

	// Make output order predictable
	var keys []string
	for u := range *we.Changes {
		keys = append(keys, u)
	}
	sort.Strings(keys)
	changeCount := len(keys)

	for _, u := range keys {
		edits := (*we.Changes)[u]
		uri := span.NewURI(u)
		cmdFile := conn.AddFile(ctx, uri)
		filename := cmdFile.uri.Filename()

		// convert LSP-style edits to []diff.TextEdit cuz Spans are handy
		renameEdits, err := source.FromProtocolEdits(cmdFile.mapper, edits)
		if err != nil {
			return errors.Errorf("%v: %v", edits, err)
		}

		newContent := diff.ApplyEdits(string(cmdFile.mapper.Content), renameEdits)

		switch {
		case r.Write:
			fmt.Fprintln(os.Stderr, filename)
			err := os.Rename(filename, filename+".orig")
			if err != nil {
				return errors.Errorf("%v: %v", edits, err)
			}
			ioutil.WriteFile(filename, []byte(newContent), 0644)
		case r.Diff:
			diffs := diff.ToUnified(filename+".orig", filename, string(cmdFile.mapper.Content), renameEdits)
			fmt.Print(diffs)
		default:
			if len(keys) > 1 {
				fmt.Printf("%s:\n", filepath.Base(filename))
			}
			fmt.Print(string(newContent))
			if changeCount > 1 { // if this wasn't last change, print newline
				fmt.Println()
			}
			changeCount -= 1
		}
	}
	return nil
}
