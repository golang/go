// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/tool"
)

// A Definition is the result of a 'definition' query.
type Definition struct {
	Span        span.Span `json:"span"`        // span of the definition
	Description string    `json:"description"` // description of the denoted object
}

// These constant is printed in the help, and then used in a test to verify the
// help is still valid.
// They refer to "Set" in "flag.FlagSet" from the DetailedHelp method below.
const (
	exampleLine   = 44
	exampleColumn = 47
	exampleOffset = 1270
)

// definition implements the definition verb for gopls.
type definition struct {
	app *Application

	JSON              bool `flag:"json" help:"emit output in JSON format"`
	MarkdownSupported bool `flag:"markdown" help:"support markdown in responses"`
}

func (d *definition) Name() string      { return "definition" }
func (d *definition) Parent() string    { return d.app.Name() }
func (d *definition) Usage() string     { return "[definition-flags] <position>" }
func (d *definition) ShortHelp() string { return "show declaration of selected identifier" }
func (d *definition) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprintf(f.Output(), `
Example: show the definition of the identifier at syntax at offset %[1]v in this file (flag.FlagSet):

	$ gopls definition internal/lsp/cmd/definition.go:%[1]v:%[2]v
	$ gopls definition internal/lsp/cmd/definition.go:#%[3]v

definition-flags:
`, exampleLine, exampleColumn, exampleOffset)
	printFlagDefaults(f)
}

// Run performs the definition query as specified by args and prints the
// results to stdout.
func (d *definition) Run(ctx context.Context, args ...string) error {
	if len(args) != 1 {
		return tool.CommandLineErrorf("definition expects 1 argument")
	}
	// Plaintext makes more sense for the command line.
	opts := d.app.options
	d.app.options = func(o *source.Options) {
		if opts != nil {
			opts(o)
		}
		o.PreferredContentFormat = protocol.PlainText
		if d.MarkdownSupported {
			o.PreferredContentFormat = protocol.Markdown
		}
	}
	conn, err := d.app.connect(ctx)
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
	tdpp := protocol.TextDocumentPositionParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
		Position:     loc.Range.Start,
	}
	p := protocol.DefinitionParams{
		TextDocumentPositionParams: tdpp,
	}
	locs, err := conn.Definition(ctx, &p)
	if err != nil {
		return fmt.Errorf("%v: %v", from, err)
	}

	if len(locs) == 0 {
		return fmt.Errorf("%v: not an identifier", from)
	}
	q := protocol.HoverParams{
		TextDocumentPositionParams: tdpp,
	}
	hover, err := conn.Hover(ctx, &q)
	if err != nil {
		return fmt.Errorf("%v: %v", from, err)
	}
	if hover == nil {
		return fmt.Errorf("%v: not an identifier", from)
	}
	file = conn.AddFile(ctx, fileURI(locs[0].URI))
	if file.err != nil {
		return fmt.Errorf("%v: %v", from, file.err)
	}
	definition, err := file.mapper.Span(locs[0])
	if err != nil {
		return fmt.Errorf("%v: %v", from, err)
	}
	description := strings.TrimSpace(hover.Contents.Value)
	result := &Definition{
		Span:        definition,
		Description: description,
	}
	if d.JSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "\t")
		return enc.Encode(result)
	}
	fmt.Printf("%v: defined here as %s", result.Span, result.Description)
	return nil
}
