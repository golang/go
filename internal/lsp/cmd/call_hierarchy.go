// Copyright 2020 The Go Authors. All rights reserved.
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

// callHierarchy implements the callHierarchy verb for gopls
type callHierarchy struct {
	app *Application
}

func (c *callHierarchy) Name() string      { return "call_hierarchy" }
func (c *callHierarchy) Usage() string     { return "<position>" }
func (c *callHierarchy) ShortHelp() string { return "display selected identifier's call hierarchy" }
func (c *callHierarchy) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Example:

  $ # 1-indexed location (:line:column or :#offset) of the target identifier
  $ gopls call_hierarchy helper/helper.go:8:6
  $ gopls call_hierarchy helper/helper.go:#53

  gopls call_hierarchy flags are:
`)
	f.PrintDefaults()
}

func (c *callHierarchy) Run(ctx context.Context, args ...string) error {
	if len(args) != 1 {
		return tool.CommandLineErrorf("call_hierarchy expects 1 argument (position)")
	}

	conn, err := c.app.connect(ctx)
	if err != nil {
		return err
	}
	defer conn.terminate(ctx)

	from := span.Parse(args[0])
	file := conn.AddFile(ctx, from.URI())
	if file.err != nil {
		return file.err
	}

	columnMapper := file.mapper
	loc, err := columnMapper.Location(from)
	if err != nil {
		return err
	}

	p := protocol.CallHierarchyPrepareParams{
		TextDocumentPositionParams: protocol.TextDocumentPositionParams{
			TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
			Position:     loc.Range.Start,
		},
	}

	callItems, err := conn.PrepareCallHierarchy(ctx, &p)
	if err != nil {
		return err
	}
	if len(callItems) == 0 {
		return fmt.Errorf("function declaration identifier not found at %v", args[0])
	}

	for _, item := range callItems {
		incomingCalls, err := conn.IncomingCalls(ctx, &protocol.CallHierarchyIncomingCallsParams{Item: item})
		if err != nil {
			return err
		}
		for i, call := range incomingCalls {
			printString, err := toPrintString(columnMapper, call.From)
			if err != nil {
				return err
			}
			fmt.Printf("caller[%d]: %s\n", i, printString)
		}

		printString, err := toPrintString(columnMapper, item)
		if err != nil {
			return err
		}
		fmt.Printf("identifier: %s\n", printString)

		outgoingCalls, err := conn.OutgoingCalls(ctx, &protocol.CallHierarchyOutgoingCallsParams{Item: item})
		if err != nil {
			return err
		}
		for i, call := range outgoingCalls {
			printString, err := toPrintString(columnMapper, call.To)
			if err != nil {
				return err
			}
			fmt.Printf("callee[%d]: %s\n", i, printString)
		}
	}

	return nil
}

func toPrintString(mapper *protocol.ColumnMapper, item protocol.CallHierarchyItem) (string, error) {
	span, err := mapper.Span(protocol.Location{URI: item.URI, Range: item.Range})
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%v %v at %v", item.Detail, item.Name, span), nil
}
