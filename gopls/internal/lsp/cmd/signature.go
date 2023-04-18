// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmd

import (
	"context"
	"flag"
	"fmt"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/tool"
)

// signature implements the signature verb for gopls
type signature struct {
	app *Application
}

func (r *signature) Name() string      { return "signature" }
func (r *signature) Parent() string    { return r.app.Name() }
func (r *signature) Usage() string     { return "<position>" }
func (r *signature) ShortHelp() string { return "display selected identifier's signature" }
func (r *signature) DetailedHelp(f *flag.FlagSet) {
	fmt.Fprint(f.Output(), `
Example:

	$ # 1-indexed location (:line:column or :#offset) of the target identifier
	$ gopls signature helper/helper.go:8:6
	$ gopls signature helper/helper.go:#53
`)
	printFlagDefaults(f)
}

func (r *signature) Run(ctx context.Context, args ...string) error {
	if len(args) != 1 {
		return tool.CommandLineErrorf("signature expects 1 argument (position)")
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

	p := protocol.SignatureHelpParams{
		TextDocumentPositionParams: protocol.LocationTextDocumentPositionParams(loc),
	}

	s, err := conn.SignatureHelp(ctx, &p)
	if err != nil {
		return err
	}

	if s == nil || len(s.Signatures) == 0 {
		return tool.CommandLineErrorf("%v: not a function", from)
	}

	// there is only ever one possible signature,
	// see toProtocolSignatureHelp in lsp/signature_help.go
	signature := s.Signatures[0]
	fmt.Printf("%s\n", signature.Label)
	switch x := signature.Documentation.Value.(type) {
	case string:
		if x != "" {
			fmt.Printf("\n%s\n", x)
		}
	case protocol.MarkupContent:
		if x.Value != "" {
			fmt.Printf("\n%s\n", x.Value)
		}
	}

	return nil
}
