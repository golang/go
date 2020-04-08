// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"flag"
	"fmt"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/tool"
)

// workspaceSymbol implements the workspace_symbol verb for gopls.
type workspaceSymbol struct {
	Matcher string `flag:"matcher" help:"specifies the type of matcher: fuzzy, caseSensitive, or caseInsensitive.\nThe default is caseInsensitive."`

	app *Application
}

func (r *workspaceSymbol) Name() string      { return "workspace_symbol" }
func (r *workspaceSymbol) Usage() string     { return "<query>" }
func (r *workspaceSymbol) ShortHelp() string { return "search symbols in workspace" }
func (r *workspaceSymbol) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Example:

  $ gopls workspace_symbol -matcher fuzzy 'wsymbols'

gopls workspace_symbol flags are:
`)
	f.PrintDefaults()
}

func (r *workspaceSymbol) Run(ctx context.Context, args ...string) error {
	if len(args) != 1 {
		return tool.CommandLineErrorf("workspace_symbol expects 1 argument")
	}

	opts := r.app.options
	r.app.options = func(o *source.Options) {
		if opts != nil {
			opts(o)
		}
		switch r.Matcher {
		case "fuzzy":
			o.SymbolMatcher = source.SymbolFuzzy
		case "caseSensitive":
			o.SymbolMatcher = source.SymbolCaseSensitive
		default:
			o.SymbolMatcher = source.SymbolCaseInsensitive
		}
	}

	conn, err := r.app.connect(ctx)
	if err != nil {
		return err
	}
	defer conn.terminate(ctx)

	p := protocol.WorkspaceSymbolParams{
		Query: args[0],
	}

	symbols, err := conn.Symbol(ctx, &p)
	if err != nil {
		return err
	}
	for _, s := range symbols {
		f := conn.AddFile(ctx, fileURI(s.Location.URI))
		span, err := f.mapper.Span(s.Location)
		if err != nil {
			return err
		}
		fmt.Printf("%s %s %s\n", span, s.Name, s.Kind)
	}

	return nil
}
