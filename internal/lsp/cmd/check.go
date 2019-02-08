// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"flag"
	"fmt"
	"go/token"
	"io/ioutil"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
)

// definition implements the definition noun for the query command.
type check struct {
	app *Application
}

type checkClient struct {
	baseClient
	diagnostics chan entry
}

type entry struct {
	uri         span.URI
	diagnostics []protocol.Diagnostic
}

func (c *check) Name() string      { return "check" }
func (c *check) Usage() string     { return "<filename>" }
func (c *check) ShortHelp() string { return "show diagnostic results for the specified file" }
func (c *check) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Example: show the diagnostic results of this file:

  $ gopls check internal/lsp/cmd/check.go

	gopls check flags are:
`)
	f.PrintDefaults()
}

// Run performs the check on the files specified by args and prints the
// results to stdout.
func (c *check) Run(ctx context.Context, args ...string) error {
	if len(args) == 0 {
		// no files, so no results
		return nil
	}
	client := &checkClient{
		diagnostics: make(chan entry),
	}
	client.app = c.app
	checking := map[span.URI][]byte{}
	// now we ready to kick things off
	server, err := c.app.connect(ctx, client)
	if err != nil {
		return err
	}
	for _, arg := range args {
		uri := span.FileURI(arg)
		content, err := ioutil.ReadFile(arg)
		if err != nil {
			return err
		}
		checking[uri] = content
		p := &protocol.DidOpenTextDocumentParams{}
		p.TextDocument.URI = string(uri)
		p.TextDocument.Text = string(content)
		if err := server.DidOpen(ctx, p); err != nil {
			return err
		}
	}
	// now wait for results
	for entry := range client.diagnostics {
		//TODO:timeout?
		content, found := checking[entry.uri]
		if !found {
			continue
		}
		fset := token.NewFileSet()
		f := fset.AddFile(string(entry.uri), -1, len(content))
		f.SetLinesForContent(content)
		m := protocol.NewColumnMapper(entry.uri, fset, f, content)
		for _, d := range entry.diagnostics {
			spn, err := m.RangeSpan(d.Range)
			if err != nil {
				return fmt.Errorf("Could not convert position %v for %q", d.Range, d.Message)
			}
			fmt.Printf("%v: %v\n", spn, d.Message)
		}
		delete(checking, entry.uri)
		if len(checking) == 0 {
			return nil
		}
	}
	return fmt.Errorf("did not get all results")
}

func (c *checkClient) PublishDiagnostics(ctx context.Context, p *protocol.PublishDiagnosticsParams) error {
	c.diagnostics <- entry{
		uri:         span.URI(p.URI),
		diagnostics: p.Diagnostics,
	}
	return nil
}
