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

// implementation implements the implementation verb for gopls
type implementation struct {
	app *Application
}

func (i *implementation) Name() string      { return "implementation" }
func (i *implementation) Parent() string    { return i.app.Name() }
func (i *implementation) Usage() string     { return "<position>" }
func (i *implementation) ShortHelp() string { return "display selected identifier's implementation" }
func (i *implementation) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Example:

	$ # 1-indexed location (:line:column or :#offset) of the target identifier
	$ gopls implementation helper/helper.go:8:6
	$ gopls implementation helper/helper.go:#53
`)
	printFlagDefaults(f)
}

func (i *implementation) Run(ctx context.Context, args ...string) error {
	if len(args) != 1 {
		return tool.CommandLineErrorf("implementation expects 1 argument (position)")
	}

	conn, err := i.app.connect(ctx)
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

	p := protocol.ImplementationParams{
		TextDocumentPositionParams: protocol.TextDocumentPositionParams{
			TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
			Position:     loc.Range.Start,
		},
	}

	implementations, err := conn.Implementation(ctx, &p)
	if err != nil {
		return err
	}

	var spans []string
	for _, impl := range implementations {
		f := conn.AddFile(ctx, fileURI(impl.URI))
		span, err := f.mapper.Span(impl)
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
