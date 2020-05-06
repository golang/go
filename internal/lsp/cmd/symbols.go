// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"encoding/json"
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
		if m, ok := s.(map[string]interface{}); ok {
			s, err = mapToSymbol(m)
			if err != nil {
				return err
			}
		}
		switch t := s.(type) {
		case protocol.DocumentSymbol:
			printDocumentSymbol(t)
		case protocol.SymbolInformation:
			printSymbolInformation(t)
		}
	}
	return nil
}

func mapToSymbol(m map[string]interface{}) (interface{}, error) {
	b, err := json.Marshal(m)
	if err != nil {
		return nil, err
	}

	if _, ok := m["selectionRange"]; ok {
		var s protocol.DocumentSymbol
		if err := json.Unmarshal(b, &s); err != nil {
			return nil, err
		}
		return s, nil
	}

	var s protocol.SymbolInformation
	if err := json.Unmarshal(b, &s); err != nil {
		return nil, err
	}
	return s, nil
}

func printDocumentSymbol(s protocol.DocumentSymbol) {
	fmt.Printf("%s %s %s\n", s.Name, s.Kind, positionToString(s.SelectionRange))
	// Sort children for consistency
	sort.Slice(s.Children, func(i, j int) bool {
		return s.Children[i].Name < s.Children[j].Name
	})
	for _, c := range s.Children {
		fmt.Printf("\t%s %s %s\n", c.Name, c.Kind, positionToString(c.SelectionRange))
	}
}

func printSymbolInformation(s protocol.SymbolInformation) {
	fmt.Printf("%s %s %s\n", s.Name, s.Kind, positionToString(s.Location.Range))
}

func positionToString(r protocol.Range) string {
	return fmt.Sprintf("%v:%v-%v:%v",
		r.Start.Line+1,
		r.Start.Character+1,
		r.End.Line+1,
		r.End.Character+1,
	)
}
