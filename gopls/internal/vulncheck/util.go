// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package vulncheck

import (
	"bytes"
	"fmt"
	"go/token"
	"os"
	"os/exec"

	gvc "golang.org/x/tools/gopls/internal/govulncheck"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/vuln/osv"
	"golang.org/x/vuln/vulncheck"
)

func toCallStack(src vulncheck.CallStack) CallStack {
	var dest []StackEntry
	for _, e := range src {
		dest = append(dest, toStackEntry(e))
	}
	return dest
}

func toStackEntry(src vulncheck.StackEntry) StackEntry {
	f, call := src.Function, src.Call
	pos := f.Pos
	desc := gvc.FuncName(f)
	if src.Call != nil {
		pos = src.Call.Pos // Exact call site position is helpful.
		if !call.Resolved {
			// In case of a statically unresolved call site, communicate to the client
			// that this was approximately resolved to f

			desc += " [approx.]"
		}
	}
	return StackEntry{
		Name: desc,
		URI:  filenameToURI(pos),
		Pos:  posToPosition(pos),
	}
}

// href returns a URL embedded in the entry if any.
// If no suitable URL is found, it returns a default entry in
// pkg.go.dev/vuln.
func href(vuln *osv.Entry) string {
	for _, affected := range vuln.Affected {
		if url := affected.DatabaseSpecific.URL; url != "" {
			return url
		}
	}
	for _, r := range vuln.References {
		if r.Type == "WEB" {
			return r.URL
		}
	}
	return fmt.Sprintf("https://pkg.go.dev/vuln/%s", vuln.ID)
}

func filenameToURI(pos *token.Position) protocol.DocumentURI {
	if pos == nil || pos.Filename == "" {
		return ""
	}
	return protocol.URIFromPath(pos.Filename)
}

func posToPosition(pos *token.Position) (p protocol.Position) {
	// token.Position.Line starts from 1, and
	// LSP protocol's position line is 0-based.
	if pos != nil {
		p.Line = uint32(pos.Line - 1)
		// TODO(hyangah): LSP uses UTF16 column.
		// We need utility like span.ToUTF16Column,
		// but somthing that does not require file contents.
	}
	return p
}

func goVersion() string {
	if v := os.Getenv("GOVERSION"); v != "" {
		// Unlikely to happen in practice, mostly used for testing.
		return v
	}
	out, err := exec.Command("go", "env", "GOVERSION").Output()
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to determine go version; skipping stdlib scanning: %v\n", err)
		return ""
	}
	return string(bytes.TrimSpace(out))
}
