// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"fmt"
	"io"
	"os/exec"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

// This writes the version and environment information to a writer.
func PrintVersionInfo(w io.Writer, verbose bool, markdown bool) {
	if !verbose {
		printBuildInfo(w, false)
		return
	}
	fmt.Fprint(w, "#### Build info\n\n")
	if markdown {
		fmt.Fprint(w, "```\n")
	}
	printBuildInfo(w, true)
	fmt.Fprint(w, "\n")
	if markdown {
		fmt.Fprint(w, "```\n")
	}
	fmt.Fprint(w, "\n#### Go info\n\n")
	if markdown {
		fmt.Fprint(w, "```\n")
	}
	cmd := exec.Command("go", "version")
	cmd.Stdout = w
	cmd.Run()
	fmt.Fprint(w, "\n")
	cmd = exec.Command("go", "env")
	cmd.Stdout = w
	cmd.Run()
	if markdown {
		fmt.Fprint(w, "```\n")
	}
}

func getSourceFile(ctx context.Context, v source.View, uri span.URI) (source.File, *protocol.ColumnMapper, error) {
	f, err := v.GetFile(ctx, uri)
	if err != nil {
		return nil, nil, err
	}
	filename, err := f.URI().Filename()
	if err != nil {
		return nil, nil, err
	}
	fc := f.Content(ctx)
	if fc.Error != nil {
		return nil, nil, fc.Error
	}
	m := protocol.NewColumnMapper(f.URI(), filename, f.FileSet(), f.GetToken(ctx), fc.Data)

	return f, m, nil
}

func getGoFile(ctx context.Context, v source.View, uri span.URI) (source.GoFile, *protocol.ColumnMapper, error) {
	f, m, err := getSourceFile(ctx, v, uri)
	if err != nil {
		return nil, nil, err
	}
	gof, ok := f.(source.GoFile)
	if !ok {
		return nil, nil, fmt.Errorf("not a Go file %v", f.URI())
	}
	return gof, m, nil
}
