// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hints

import (
	"encoding/json"
	"fmt"
	"net/url"
	"path/filepath"
)

const SchemaVersion = "1.0"

const (
	SiteTry         = "try"
	SiteConstFor    = "const_for"
	SiteConstSwitch = "const_switch"
	SiteInterp      = "interp"
)

const (
	MethodInlayHint = "textDocument/inlayHint"
	MethodHover     = "textDocument/hover"
)

type Report struct {
	Version string `json:"schemaVersion"`
	Package string `json:"package"`
	Files   []File `json:"files"`
}

type File struct {
	File     string     `json:"file"`
	Narrowed []Narrowed `json:"narrowed"`
	Unions   []Union    `json:"unions"`
	Sites    []Site     `json:"sites"`
	Allocs   []Alloc    `json:"allocs"`
}

type Narrowed struct {
	Position Position `json:"position"`
	Name     string   `json:"name"`
	Type     string   `json:"type"`
}

type Union struct {
	Position Position `json:"position"`
	Type     string   `json:"type"`
	Members  []string `json:"members"`
}

type Site struct {
	Range      Range  `json:"range"`
	Kind       string `json:"kind"`
	Lowered    string `json:"lowered"`
	FixedBytes *int   `json:"fixedBytes,omitempty"`
}

type Alloc struct {
	Position Position `json:"position"`
	Tier     string   `json:"tier"`
	Message  string   `json:"message"`
	Reason   string   `json:"reason,omitempty"`
}

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

func (r *Report) file(name string) *File {
	for i := range r.Files {
		if filepath.Base(r.Files[i].File) == filepath.Base(name) {
			return &r.Files[i]
		}
	}
	return nil
}

type request struct {
	TextDocument struct {
		URI string `json:"uri"`
	} `json:"textDocument"`
	Range    Range    `json:"range"`
	Position Position `json:"position"`
}

func Serve(method string, params, report json.RawMessage) (json.RawMessage, error) {
	switch method {
	case MethodInlayHint:
		return serveInlayHint(params, report)
	case MethodHover:
		return serveHover(params, report)
	}
	return nil, fmt.Errorf("hints: no hints for method %q", method)
}

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
