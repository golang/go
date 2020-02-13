// Copyright 2019 The Go Authors. All rights reserved.
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

// prepareRename implements the prepare_rename verb for gopls.
type prepareRename struct {
	app *Application
}

func (r *prepareRename) Name() string      { return "prepare_rename" }
func (r *prepareRename) Usage() string     { return "<position>" }
func (r *prepareRename) ShortHelp() string { return "test validity of a rename operation at location" }
func (r *prepareRename) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Example:

	$ # 1-indexed location (:line:column or :#offset) of the target identifier
	$ gopls prepare_rename helper/helper.go:8:6
	$ gopls prepare_rename helper/helper.go:#53

	gopls prepare_rename flags are:
`)
	f.PrintDefaults()
}

func (r *prepareRename) Run(ctx context.Context, args ...string) error {
	if len(args) != 1 {
		return tool.CommandLineErrorf("prepare_rename expects 1 argument (file)")
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
	p := protocol.PrepareRenameParams{
		TextDocumentPositionParams: protocol.TextDocumentPositionParams{
			TextDocument: protocol.TextDocumentIdentifier{URI: loc.URI},
			Position:     loc.Range.Start,
		},
	}
	result, err := conn.PrepareRename(ctx, &p)
	if err != nil {
		return fmt.Errorf("prepare_rename failed: %v", err)
	}
	if result == nil {
		return fmt.Errorf("request is not valid at the given position")
	}

	l := protocol.Location{Range: *result}
	s, err := file.mapper.Span(l)
	if err != nil {
		return err
	}

	fmt.Println(s)
	return nil
}
