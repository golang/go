// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"flag"
	"fmt"
	"strings"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/tool"
)

// callHierarchy implements the callHierarchy verb for gopls.
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

	loc, err := file.mapper.Location(from)
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
			// From the spec: CallHierarchyIncomingCall.FromRanges is relative to
			// the caller denoted by CallHierarchyIncomingCall.from.
			printString, err := callItemPrintString(ctx, conn, call.From, call.From.URI, call.FromRanges)
			if err != nil {
				return err
			}
			fmt.Printf("caller[%d]: %s\n", i, printString)
		}

		printString, err := callItemPrintString(ctx, conn, item, "", nil)
		if err != nil {
			return err
		}
		fmt.Printf("identifier: %s\n", printString)

		outgoingCalls, err := conn.OutgoingCalls(ctx, &protocol.CallHierarchyOutgoingCallsParams{Item: item})
		if err != nil {
			return err
		}
		for i, call := range outgoingCalls {
			// From the spec: CallHierarchyOutgoingCall.FromRanges is the range
			// relative to the caller, e.g the item passed to
			printString, err := callItemPrintString(ctx, conn, call.To, item.URI, call.FromRanges)
			if err != nil {
				return err
			}
			fmt.Printf("callee[%d]: %s\n", i, printString)
		}
	}

	return nil
}

// callItemPrintString returns a protocol.CallHierarchyItem object represented as a string.
// item and call ranges (protocol.Range) are converted to user friendly spans (1-indexed).
func callItemPrintString(ctx context.Context, conn *connection, item protocol.CallHierarchyItem, callsURI protocol.DocumentURI, calls []protocol.Range) (string, error) {
	itemFile := conn.AddFile(ctx, item.URI.SpanURI())
	if itemFile.err != nil {
		return "", itemFile.err
	}
	itemSpan, err := itemFile.mapper.Span(protocol.Location{URI: item.URI, Range: item.Range})
	if err != nil {
		return "", err
	}

	callsFile := conn.AddFile(ctx, callsURI.SpanURI())
	if callsURI != "" && callsFile.err != nil {
		return "", callsFile.err
	}
	var callRanges []string
	for _, rng := range calls {
		callSpan, err := callsFile.mapper.Span(protocol.Location{URI: item.URI, Range: rng})
		if err != nil {
			return "", err
		}

		spn := fmt.Sprint(callSpan)
		callRanges = append(callRanges, fmt.Sprint(spn[strings.Index(spn, ":")+1:]))
	}

	printString := fmt.Sprintf("function %s in %v", item.Name, itemSpan)
	if len(calls) > 0 {
		printString = fmt.Sprintf("ranges %s in %s from/to %s", strings.Join(callRanges, ", "), callsURI.SpanURI().Filename(), printString)
	}
	return printString, nil
}
