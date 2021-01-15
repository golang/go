// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"flag"
	"fmt"

	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

// check implements the check verb for gopls.
type check struct {
	app *Application
}

func (c *check) Name() string      { return "check" }
func (c *check) Usage() string     { return "<filename>" }
func (c *check) ShortHelp() string { return "show diagnostic results for the specified file" }
func (c *check) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Example: show the diagnostic results of this file:

  $ gopls check internal/lsp/cmd/check.go
`)
	f.PrintDefaults()
}

// Run performs the check on the files specified by args and prints the
// results to stdout.
func (c *check) Run(ctx context.Context, args ...string) error {
	if len(args) == 0 {
		// no files, so no results
		return nil
	}
	checking := map[span.URI]*cmdFile{}
	var uris []span.URI
	// now we ready to kick things off
	conn, err := c.app.connect(ctx)
	if err != nil {
		return err
	}
	defer conn.terminate(ctx)
	for _, arg := range args {
		uri := span.URIFromPath(arg)
		uris = append(uris, uri)
		file := conn.AddFile(ctx, uri)
		if file.err != nil {
			return file.err
		}
		checking[uri] = file
	}
	if err := conn.diagnoseFiles(ctx, uris); err != nil {
		return err
	}
	conn.Client.filesMu.Lock()
	defer conn.Client.filesMu.Unlock()

	for _, file := range checking {
		for _, d := range file.diagnostics {
			spn, err := file.mapper.RangeSpan(d.Range)
			if err != nil {
				return errors.Errorf("Could not convert position %v for %q", d.Range, d.Message)
			}
			fmt.Printf("%v: %v\n", spn, d.Message)
		}
	}
	return nil
}
