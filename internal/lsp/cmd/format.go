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
	// now we ready to kick things off
	conn, err := f.app.connect(ctx)
	if err != nil {
		return err
	}
	defer conn.terminate(ctx)
	for _, arg := range args {
		spn := span.Parse(arg)
		file := conn.AddFile(ctx, spn.URI())
		if file.err != nil {
			return file.err
		}
		filename, _ := spn.URI().Filename() // this cannot fail, already checked in AddFile above
		loc, err := file.mapper.Location(spn)
		if err != nil {
			return err
		}
		if loc.Range.Start != loc.Range.End {
			return fmt.Errorf("only full file formatting supported")
		}
		p := protocol.DocumentFormattingParams{
			TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
		}
		edits, err := conn.Formatting(ctx, &p)
		if err != nil {
			return fmt.Errorf("%v: %v", spn, err)
		}
		sedits, err := lsp.FromProtocolEdits(file.mapper, edits)
		if err != nil {
			return fmt.Errorf("%v: %v", spn, err)
		}
		ops := source.EditsToDiff(sedits)
		lines := diff.SplitLines(string(file.mapper.Content))
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
			u := diff.ToUnified(filename+".orig", filename, lines, ops)
			fmt.Print(u)
		}
		if printIt {
			fmt.Print(formatted)
		}
	}
	return nil
}
