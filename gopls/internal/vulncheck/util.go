// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package vulncheck

import (
	"fmt"
	"go/token"
	"strings"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/vuln/osv"
	"golang.org/x/vuln/vulncheck"
)

// fixedVersion returns the semantic version of the module
// version with a fix. The semantic version is
// as defined by SemVer 2.0.0, with no leading “v” prefix.
// Returns an empty string if there is no reported fix.
func fixedVersion(info *osv.Entry) string {
	var fixed string
	for _, a := range info.Affected {
		for _, r := range a.Ranges {
			if r.Type != "SEMVER" {
				continue
			}
			for _, e := range r.Events {
				if e.Fixed != "" {
					// assuming the later entry has higher semver.
					// TODO: check assumption.
					fixed = "v" + e.Fixed
				}
			}
		}
	}
	return fixed
}

const maxNumCallStacks = 64

func toCallStacks(src []vulncheck.CallStack) []CallStack {
	if len(src) > maxNumCallStacks {
		src = src[:maxNumCallStacks]
	}
	var dest []CallStack
	for _, s := range src {
		dest = append(dest, toCallStack(s))
	}
	return dest
}

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
	desc := funcName(f)
	if src.Call != nil {
		pos = src.Call.Pos
		desc = funcNameInCallSite(call) + " called from " + desc
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

func funcName(fn *vulncheck.FuncNode) string {
	return strings.TrimPrefix(fn.String(), "*")
}

func funcNameInCallSite(call *vulncheck.CallSite) string {
	if call.RecvType == "" {
		return call.Name
	}
	return fmt.Sprintf("%s.%s", call.RecvType, call.Name)
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
