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

	guru "golang.org/x/tools/cmd/guru/serial"
	"golang.org/x/tools/internal/lsp/protocol"
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

// definition implements the definition noun for the query command.
type definition struct {
	query *query
}

func (d *definition) Name() string      { return "definition" }
func (d *definition) Usage() string     { return "<position>" }
func (d *definition) ShortHelp() string { return "show declaration of selected identifier" }
func (d *definition) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprintf(f.Output(), `
Example: show the definition of the identifier at syntax at offset %[1]v in this file (flag.FlagSet):

$ gopls definition internal/lsp/cmd/definition.go:%[1]v:%[2]v
$ gopls definition internal/lsp/cmd/definition.go:#%[3]v

	gopls query definition flags are:
`, exampleLine, exampleColumn, exampleOffset)
	f.PrintDefaults()
}

// Run performs the definition query as specified by args and prints the
// results to stdout.
func (d *definition) Run(ctx context.Context, args ...string) error {
	if len(args) != 1 {
		return tool.CommandLineErrorf("definition expects 1 argument")
	}
	conn, err := d.query.app.connect(ctx)
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
	p := protocol.TextDocumentPositionParams{
		TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
		Position:     loc.Range.Start,
	}
	locs, err := conn.Definition(ctx, &p)
	if err != nil {
		return fmt.Errorf("%v: %v", from, err)
	}

	if len(locs) == 0 {
		return fmt.Errorf("%v: not an identifier", from)
	}
	hover, err := conn.Hover(ctx, &p)
	if err != nil {
		return fmt.Errorf("%v: %v", from, err)
	}
	if hover == nil {
		return fmt.Errorf("%v: not an identifier", from)
	}
	file = conn.AddFile(ctx, span.NewURI(locs[0].URI))
	if file.err != nil {
		return fmt.Errorf("%v: %v", from, file.err)
	}
	definition, err := file.mapper.Span(locs[0])
	if err != nil {
		return fmt.Errorf("%v: %v", from, err)
	}
	description := strings.TrimSpace(hover.Contents.Value)
	var result interface{}
	switch d.query.Emulate {
	case "":
		result = &Definition{
			Span:        definition,
			Description: description,
		}
	case emulateGuru:
		pos := span.New(definition.URI(), definition.Start(), definition.Start())
		result = &guru.Definition{
			ObjPos: fmt.Sprint(pos),
			Desc:   description,
		}
	default:
		return fmt.Errorf("unknown emulation for definition: %s", d.query.Emulate)
	}
	if err != nil {
		return err
	}
	if d.query.JSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "\t")
		return enc.Encode(result)
	}
	switch d := result.(type) {
	case *Definition:
		fmt.Printf("%v: defined here as %s", d.Span, d.Description)
	case *guru.Definition:
		fmt.Printf("%s: defined here as %s", d.ObjPos, d.Desc)
	default:
		return fmt.Errorf("no printer for type %T", result)
	}
	return nil
}
