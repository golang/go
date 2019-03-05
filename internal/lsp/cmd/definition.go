// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"go/types"
	"os"

	guru "golang.org/x/tools/cmd/guru/serial"
	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/tool"
)

// A Definition is the result of a 'definition' query.
type Definition struct {
	Location    Location `json:"location"`    // location of the definition
	Description string   `json:"description"` // description of the denoted object
}

// This constant is printed in the help, and then used in a test to verify the
// help is still valid.
// It should be the byte offset in this file of the "Set" in "flag.FlagSet" from
// the DetailedHelp method below.
const exampleOffset = 1277

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

  $ gopls definition internal/lsp/cmd/definition.go:#%[1]v

	gopls definition flags are:
`, exampleOffset)
	f.PrintDefaults()
}

// Run performs the definition query as specified by args and prints the
// results to stdout.
func (d *definition) Run(ctx context.Context, args ...string) error {
	if len(args) != 1 {
		return tool.CommandLineErrorf("definition expects 1 argument")
	}
	view := cache.NewView(&d.query.app.Config)
	from, err := parseLocation(args[0])
	if err != nil {
		return err
	}
	f, err := view.GetFile(ctx, source.ToURI(from.Filename))
	if err != nil {
		return err
	}
	tok := f.GetToken(ctx)
	pos := tok.Pos(from.Start.Offset)
	if !pos.IsValid() {
		return fmt.Errorf("invalid position %v", from.Start.Offset)
	}
	ident, err := source.Identifier(ctx, view, f, pos)
	if err != nil {
		return err
	}
	if ident == nil {
		return fmt.Errorf("not an identifier")
	}
	var result interface{}
	switch d.query.Emulate {
	case "":
		result, err = buildDefinition(ctx, view, ident)
	case emulateGuru:
		result, err = buildGuruDefinition(ctx, view, ident)
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
		fmt.Printf("%v: defined here as %s", d.Location, d.Description)
	case *guru.Definition:
		fmt.Printf("%s: defined here as %s", d.ObjPos, d.Desc)
	default:
		return fmt.Errorf("no printer for type %T", result)
	}
	return nil
}

func buildDefinition(ctx context.Context, view source.View, ident *source.IdentifierInfo) (*Definition, error) {
	content, err := ident.Hover(ctx, nil)
	if err != nil {
		return nil, err
	}
	return &Definition{
		Location:    newLocation(view.FileSet(), ident.Declaration.Range),
		Description: content,
	}, nil
}

func buildGuruDefinition(ctx context.Context, view source.View, ident *source.IdentifierInfo) (*guru.Definition, error) {
	loc := newLocation(view.FileSet(), ident.Declaration.Range)
	pkg := ident.File.GetPackage(ctx)
	// guru does not support ranges
	loc.End = loc.Start
	// Behavior that attempts to match the expected output for guru. For an example
	// of the format, see the associated definition tests.
	buf := &bytes.Buffer{}
	q := types.RelativeTo(pkg.Types)
	qualifyName := ident.Declaration.Object.Pkg() != pkg.Types
	name := ident.Name
	var suffix interface{}
	switch obj := ident.Declaration.Object.(type) {
	case *types.TypeName:
		fmt.Fprint(buf, "type")
	case *types.Var:
		if obj.IsField() {
			qualifyName = false
			fmt.Fprint(buf, "field")
			suffix = obj.Type()
		} else {
			fmt.Fprint(buf, "var")
		}
	case *types.Func:
		fmt.Fprint(buf, "func")
		typ := obj.Type()
		if obj.Type() != nil {
			if sig, ok := typ.(*types.Signature); ok {
				buf := &bytes.Buffer{}
				if recv := sig.Recv(); recv != nil {
					if named, ok := recv.Type().(*types.Named); ok {
						fmt.Fprintf(buf, "(%s).%s", named.Obj().Name(), name)
					}
				}
				if buf.Len() == 0 {
					buf.WriteString(name)
				}
				types.WriteSignature(buf, sig, q)
				name = buf.String()
			}
		}
	default:
		fmt.Fprintf(buf, "unknown [%T]", obj)
	}
	fmt.Fprint(buf, " ")
	if qualifyName {
		fmt.Fprintf(buf, "%s.", ident.Declaration.Object.Pkg().Path())
	}
	fmt.Fprint(buf, name)
	if suffix != nil {
		fmt.Fprint(buf, " ")
		fmt.Fprint(buf, suffix)
	}
	return &guru.Definition{
		ObjPos: fmt.Sprint(loc),
		Desc:   buf.String(),
	}, nil
}
