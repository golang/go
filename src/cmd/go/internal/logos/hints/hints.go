// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package hints renders the editor features of the loGo/s language server
// that the compiler computes (SPEC Sec 21.2 to 21.4): inlay hints for
// narrowed types, union members, and allocation tiers; hovers that show
// lowered code and the fixed byte count of an interpolation; and code
// actions that apply the autofix candidates of the JSON diagnostics.
//
// The input of the first two is the report that the compiler writes under
// -logos.mode=hints -logos.hintsout=FILE for the original sources of one
// package. Its positions are already protocol positions, 0-based line and
// UTF-16 character, so the renderers need no document text. Code actions
// need the text, because JSON diagnostics carry byte columns.
//
// The language-server proxy wires Serve into its request path as the hints
// function of its options, so that inlay hints and hovers of .lgo and .go
// files come from the report and reach the client next to the answer of
// gopls.
package hints

import (
	"encoding/json"
	"fmt"
	"net/url"
	"path/filepath"
)

// SchemaVersion is the version of the report this package reads.
const SchemaVersion = "1.0"

// Allocation tiers of Alloc.
const (
	TierStack  = "stack"
	TierArena  = "arena"
	TierHeap   = "heap"
	TierRodata = "rodata"
)

// Site kinds of Site.
const (
	SiteTry         = "try"
	SiteConstFor    = "const_for"
	SiteConstSwitch = "const_switch"
	SiteInterp      = "interp"
)

// Methods that Serve answers.
const (
	MethodInlayHint = "textDocument/inlayHint"
	MethodHover     = "textDocument/hover"
)

// Report is the hints report of one package.
type Report struct {
	Version string `json:"schemaVersion"`
	Package string `json:"package"`
	Files   []File `json:"files"`
}

// File holds the hints of one source file, named as the compiler was given
// the file.
type File struct {
	File     string     `json:"file"`
	Narrowed []Narrowed `json:"narrowed"`
	Unions   []Union    `json:"unions"`
	Sites    []Site     `json:"sites"`
	Allocs   []Alloc    `json:"allocs"`
}

// Narrowed is a use of an option variable that the checker narrowed.
type Narrowed struct {
	Position Position `json:"position"`
	Name     string   `json:"name"`
	Type     string   `json:"type"`
}

// Union is the member list of a closed union at a type switch.
type Union struct {
	Position Position `json:"position"`
	Type     string   `json:"type"`
	Members  []string `json:"members"`
}

// Site is a construct that lowering rewrites. FixedBytes is set for
// interpolations only.
type Site struct {
	Range      Range  `json:"range"`
	Kind       string `json:"kind"`
	Lowered    string `json:"lowered"`
	FixedBytes *int   `json:"fixedBytes,omitempty"`
}

// Alloc is one allocation decision: its tier and the -m diagnostic text.
type Alloc struct {
	Position Position `json:"position"`
	Tier     string   `json:"tier"`
	Message  string   `json:"message"`
	Reason   string   `json:"reason,omitempty"`
}

// Decode parses a report and rejects one of another schema version.
func Decode(data []byte) (*Report, error) {
	r := new(Report)
	if err := json.Unmarshal(data, r); err != nil {
		return nil, fmt.Errorf("hints: %w", err)
	}
	if r.Version != SchemaVersion {
		return nil, fmt.Errorf("hints: unsupported schema version %q (want %q)", r.Version, SchemaVersion)
	}
	return r, nil
}

// file returns the hints of the file named name. A report covers one package
// directory, so the base name identifies the file.
func (r *Report) file(name string) *File {
	for i := range r.Files {
		if filepath.Base(r.Files[i].File) == filepath.Base(name) {
			return &r.Files[i]
		}
	}
	return nil
}

// request is the part of an inlayHint or hover request that the renderers
// read: the document, and the range or the position.
type request struct {
	TextDocument struct {
		URI string `json:"uri"`
	} `json:"textDocument"`
	Range    Range    `json:"range"`
	Position Position `json:"position"`
}

// Serve answers one client request from a report. It is the entry point of
// the proxy: method is the JSON-RPC method, params its raw parameters in the
// client's coordinates, and report the raw hints report of the package that
// holds the file. An inlayHint answer is a JSON array and a hover answer is
// a Hover, or nil when no site lies under the position. Any other method is
// an error.
func Serve(method string, params, report json.RawMessage) (json.RawMessage, error) {
	switch method {
	case MethodInlayHint:
		return serveInlayHint(params, report)
	case MethodHover:
		return serveHover(params, report)
	}
	return nil, fmt.Errorf("hints: no hints for method %q", method)
}

// serveInlayHint answers textDocument/inlayHint.
func serveInlayHint(params, report json.RawMessage) (json.RawMessage, error) {
	r, req, name, err := decodeRequest(params, report)
	if err != nil {
		return nil, err
	}
	hs := InlayHints(r, name, req.Range)
	if hs == nil {
		hs = []InlayHint{}
	}
	return json.Marshal(hs)
}

// serveHover answers textDocument/hover.
func serveHover(params, report json.RawMessage) (json.RawMessage, error) {
	r, req, name, err := decodeRequest(params, report)
	if err != nil {
		return nil, err
	}
	h := HoverAt(r, name, req.Position)
	if h == nil {
		return nil, nil
	}
	return json.Marshal(h)
}

// decodeRequest parses the request parameters and the report, and returns
// them with the path of the requested file.
func decodeRequest(params, report json.RawMessage) (*Report, request, string, error) {
	var req request
	if err := json.Unmarshal(params, &req); err != nil {
		return nil, req, "", fmt.Errorf("hints: bad request parameters: %w", err)
	}
	r, err := Decode(report)
	if err != nil {
		return nil, req, "", err
	}
	u, err := url.Parse(req.TextDocument.URI)
	if err != nil {
		return nil, req, "", fmt.Errorf("hints: bad document URI: %w", err)
	}
	return r, req, u.Path, nil
}
