// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"flag"
	"fmt"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/span"
)

// format implements the format verb for gopls.
type format struct {
	EditFlags
	app *Application
}

func (c *format) Name() string      { return "format" }
func (c *format) Parent() string    { return c.app.Name() }
func (c *format) Usage() string     { return "[format-flags] <filerange>" }
func (c *format) ShortHelp() string { return "format the code according to the go standard" }
func (c *format) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
The arguments supplied may be simple file names, or ranges within files.

Example: reformat this file:

	$ gopls format -w internal/lsp/cmd/check.go

format-flags:
`)
	printFlagDefaults(f)
}

// Run performs the check on the files specified by args and prints the
// results to stdout.
func (c *format) Run(ctx context.Context, args ...string) error {
	if len(args) == 0 {
		return nil
	}
	c.app.editFlags = &c.EditFlags
	conn, err := c.app.connect(ctx, nil)
	if err != nil {
		return err
	}
	defer conn.terminate(ctx)
	for _, arg := range args {
		spn := span.Parse(arg)
		file, err := conn.openFile(ctx, spn.URI())
		if err != nil {
			return err
		}
		loc, err := file.mapper.SpanLocation(spn)
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
		if err := applyTextEdits(file.mapper, edits, c.app.editFlags); err != nil {
			return err
		}
	}
	return nil
}
