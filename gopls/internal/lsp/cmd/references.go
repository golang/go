// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"flag"
	"fmt"
	"sort"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/tool"
)

// references implements the references verb for gopls
type references struct {
	IncludeDeclaration bool `flag:"d,declaration" help:"include the declaration of the specified identifier in the results"`

	app *Application
}

func (r *references) Name() string      { return "references" }
func (r *references) Parent() string    { return r.app.Name() }
func (r *references) Usage() string     { return "[references-flags] <position>" }
func (r *references) ShortHelp() string { return "display selected identifier's references" }
func (r *references) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Example:

	$ # 1-indexed location (:line:column or :#offset) of the target identifier
	$ gopls references helper/helper.go:8:6
	$ gopls references helper/helper.go:#53

references-flags:
`)
	printFlagDefaults(f)
}

func (r *references) Run(ctx context.Context, args ...string) error {
	if len(args) != 1 {
		return tool.CommandLineErrorf("references expects 1 argument (position)")
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
	p := protocol.ReferenceParams{
		Context: protocol.ReferenceContext{
			IncludeDeclaration: r.IncludeDeclaration,
		},
		TextDocumentPositionParams: protocol.TextDocumentPositionParams{
			TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
			Position:     loc.Range.Start,
		},
	}
	locations, err := conn.References(ctx, &p)
	if err != nil {
		return err
	}
	var spans []string
	for _, l := range locations {
		f := conn.AddFile(ctx, fileURI(l.URI))
		// convert location to span for user-friendly 1-indexed line
		// and column numbers
		span, err := f.mapper.Span(l)
		if err != nil {
			return err
		}
		spans = append(spans, fmt.Sprint(span))
	}

	sort.Strings(spans)
	for _, s := range spans {
		fmt.Println(s)
	}
	return nil
}
