// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"errors"
	"flag"
	"fmt"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/tool"
)

// prepareRename implements the prepare_rename verb for gopls.
type prepareRename struct {
	app *Application
}

func (r *prepareRename) Name() string      { return "prepare_rename" }
func (r *prepareRename) Parent() string    { return r.app.Name() }
func (r *prepareRename) Usage() string     { return "<position>" }
func (r *prepareRename) ShortHelp() string { return "test validity of a rename operation at location" }
func (r *prepareRename) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Example:

	$ # 1-indexed location (:line:column or :#offset) of the target identifier
	$ gopls prepare_rename helper/helper.go:8:6
	$ gopls prepare_rename helper/helper.go:#53
`)
	printFlagDefaults(f)
}

// ErrInvalidRenamePosition is returned when prepareRename is run at a position that
// is not a candidate for renaming.
var ErrInvalidRenamePosition = errors.New("request is not valid at the given position")

func (r *prepareRename) Run(ctx context.Context, args ...string) error {
	if len(args) != 1 {
		return tool.CommandLineErrorf("prepare_rename expects 1 argument (file)")
	}

	conn, err := r.app.connect(ctx, nil)
	if err != nil {
		return err
	}
	defer conn.terminate(ctx)

	from := span.Parse(args[0])
	file, err := conn.openFile(ctx, from.URI())
	if err != nil {
		return err
	}
	loc, err := file.mapper.SpanLocation(from)
	if err != nil {
		return err
	}
	p := protocol.PrepareRenameParams{
		TextDocumentPositionParams: protocol.LocationTextDocumentPositionParams(loc),
	}
	result, err := conn.PrepareRename(ctx, &p)
	if err != nil {
		return fmt.Errorf("prepare_rename failed: %w", err)
	}
	if result == nil {
		return ErrInvalidRenamePosition
	}

	s, err := file.mapper.RangeSpan(result.Range)
	if err != nil {
		return err
	}

	fmt.Println(s)
	return nil
}
