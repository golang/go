// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"flag"
	"fmt"
	"sort"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/tool"
)

// symbols implements the symbols verb for gopls
type symbols struct {
	app *Application
}

func (r *symbols) Name() string      { return "symbols" }
func (r *symbols) Usage() string     { return "<file>" }
func (r *symbols) ShortHelp() string { return "display selected file's symbols" }
func (r *symbols) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Example:
  $ gopls symbols helper/helper.go
`)
	f.PrintDefaults()
}
func (r *symbols) Run(ctx context.Context, args ...string) error {
	if len(args) != 1 {
		return tool.CommandLineErrorf("symbols expects 1 argument (position)")
	}

	conn, err := r.app.connect(ctx)
	if err != nil {
		return err
	}
	defer conn.terminate(ctx)

	from := span.Parse(args[0])
	p := protocol.DocumentSymbolParams{
		TextDocument: protocol.TextDocumentIdentifier{
			URI: protocol.URIFromSpanURI(from.URI()),
		},
	}

	symbols, err := conn.DocumentSymbol(ctx, &p)
	if err != nil {
		return err
	}
	for _, s := range symbols {
		fmt.Println(symbolToString(s))
		// Sort children for consistency
		sort.Slice(s.Children, func(i, j int) bool {
			return s.Children[i].Name < s.Children[j].Name
		})
		for _, c := range s.Children {
			fmt.Println("\t" + symbolToString(c))
		}
	}

	return nil
}

func symbolToString(symbol protocol.DocumentSymbol) string {
	r := symbol.SelectionRange
	// convert ranges to user friendly 1-based positions
	position := fmt.Sprintf("%v:%v-%v:%v",
		r.Start.Line+1,
		r.Start.Character+1,
		r.End.Line+1,
		r.End.Character+1,
	)
	return fmt.Sprintf("%s %s %s", symbol.Name, symbol.Kind, position)
}
