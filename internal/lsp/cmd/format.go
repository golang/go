// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"flag"
	"fmt"
	"io/ioutil"
	"strings"

	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// format implements the format verb for gopls.
type format struct {
	Diff  bool `flag:"d" help:"display diffs instead of rewriting files"`
	Write bool `flag:"w" help:"write result to (source) file instead of stdout"`
	List  bool `flag:"l" help:"list files whose formatting differs from gofmt's"`

	app *Application
}

func (c *format) Name() string      { return "format" }
func (c *format) Usage() string     { return "<filerange>" }
func (c *format) ShortHelp() string { return "format the code according to the go standard" }
func (c *format) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
The arguments supplied may be simple file names, or ranges within files.

Example: reformat this file:

  $ gopls format -w internal/lsp/cmd/check.go

	gopls format flags are:
`)
	f.PrintDefaults()
}

// Run performs the check on the files specified by args and prints the
// results to stdout.
func (f *format) Run(ctx context.Context, args ...string) error {
	if len(args) == 0 {
		// no files, so no results
		return nil
	}
	client := &baseClient{}
	// now we ready to kick things off
	server, err := f.app.connect(ctx, client)
	if err != nil {
		return err
	}
	for _, arg := range args {
		spn := span.Parse(arg)
		m, err := client.AddFile(ctx, spn.URI())
		if err != nil {
			return err
		}
		filename, _ := spn.URI().Filename() // this cannot fail, already checked in AddFile above
		loc, err := m.Location(spn)
		if err != nil {
			return err
		}
		p := protocol.DocumentRangeFormattingParams{
			TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
			Range:        loc.Range,
		}
		edits, err := server.RangeFormatting(ctx, &p)
		if err != nil {
			return fmt.Errorf("%v: %v", spn, err)
		}
		sedits, err := lsp.FromProtocolEdits(m, edits)
		if err != nil {
			return fmt.Errorf("%v: %v", spn, err)
		}
		ops := source.EditsToDiff(sedits)
		lines := diff.SplitLines(string(m.Content))
		formatted := strings.Join(diff.ApplyEdits(lines, ops), "")
		printIt := true
		if f.List {
			printIt = false
			if len(edits) > 0 {
				fmt.Println(filename)
			}
		}
		if f.Write {
			printIt = false
			if len(edits) > 0 {
				ioutil.WriteFile(filename, []byte(formatted), 0644)
			}
		}
		if f.Diff {
			printIt = false
			u := diff.ToUnified(filename, filename, lines, ops)
			fmt.Print(u)
		}
		if printIt {
			fmt.Print(formatted)
		}
	}
	return nil
}
