// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"flag"
	"fmt"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/tool"
)

// foldingRanges implements the folding_ranges verb for gopls
type foldingRanges struct {
	app *Application
}

func (r *foldingRanges) Name() string      { return "folding_ranges" }
func (r *foldingRanges) Parent() string    { return r.app.Name() }
func (r *foldingRanges) Usage() string     { return "<file>" }
func (r *foldingRanges) ShortHelp() string { return "display selected file's folding ranges" }
func (r *foldingRanges) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Example:

	$ gopls folding_ranges helper/helper.go
`)
	printFlagDefaults(f)
}

func (r *foldingRanges) Run(ctx context.Context, args ...string) error {
	if len(args) != 1 {
		return tool.CommandLineErrorf("folding_ranges expects 1 argument (file)")
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

	p := protocol.FoldingRangeParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(from.URI()),
		},
	}

	ranges, err := conn.FoldingRange(ctx, &p)
	if err != nil {
		return err
	}

	for _, r := range ranges {
		fmt.Printf("%v:%v-%v:%v\n",
			r.StartLine+1,
			r.StartCharacter+1,
			r.EndLine+1,
			r.EndCharacter,
		)
	}

	return nil
}
