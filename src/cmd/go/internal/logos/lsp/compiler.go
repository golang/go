// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"encoding/json"

	"cmd/internal/logos/editmap"
)

type Compiler interface {
	Lower(ctx context.Context, dir string, buffers map[string][]byte) (*Lowering, error)

	Diagnose(ctx context.Context, dir string, buffers map[string][]byte) ([]Diagnostic, error)

	DiskChanged()
}

type Lowering struct {
	Dirs []string

	Files []LoweredFile

	Missing []string
}

type LoweredFile struct {
	Path string

	Src, Out []byte

	Map editmap.EditMap
}

type Diagnostic struct {
	SchemaVersion string `json:"schema_version"`
	Severity      string `json:"severity"`
	RuleID        string `json:"rule_id"`
	File          string `json:"file"`
	Span          struct {
		StartLine int `json:"start_line"`
		StartCol  int `json:"start_col"`
		EndLine   int `json:"end_line"`
		EndCol    int `json:"end_col"`
	} `json:"span"`
	Message string          `json:"message"`
	Raw     json.RawMessage `json:"-"`
}

func parseDiagnostic(line []byte) (Diagnostic, bool) {
	var d Diagnostic
	if err := json.Unmarshal(line, &d); err != nil || d.SchemaVersion == "" || d.File == "" {
		return Diagnostic{}, false
	}
	d.Raw = json.RawMessage(bytes.Clone(line))
	return d, true
}
