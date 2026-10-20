// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"bytes"
	"context"
	"encoding/json"
)

// Compiler is the proxy's view of the toolchain. The production
// implementation drives the logos command; tests substitute their own.
// Both methods read buffers, the unsaved texts of the open documents by
// absolute path, in place of the files on disk, and never write inside
// the workspace.
type Compiler interface {
	// Lower lowers every loGo/s file of the packages the package in dir
	// depends on, itself included, and returns the lowered texts with the
	// edit maps that relate them to the sources.
	Lower(ctx context.Context, dir string, buffers map[string][]byte) (*Lowering, error)

	// Diagnose returns the compiler and lint diagnostics of the package in
	// dir, from the logos JSON diagnostics (SPEC Sec 16).
	Diagnose(ctx context.Context, dir string, buffers map[string][]byte) ([]Diagnostic, error)
}

// Lowering is the result of lowering one package and its dependencies.
type Lowering struct {
	// Dirs lists the directory of every package of the dependency closure
	// that the workspace or a module owns, the standard library excluded.
	Dirs []string

	// Files holds one entry for each loGo/s file whose lowering succeeded.
	Files []LoweredFile

	// Missing lists the loGo/s files of the closure that produced no
	// lowered text, because their package failed to compile.
	Missing []string
}

// LoweredFile is the lowering of one loGo/s source file.
type LoweredFile struct {
	// Path is the absolute path of the loGo/s source.
	Path string

	// Src is the source text the compiler read, and Out the Go text it
	// ejected from it.
	Src, Out []byte

	// Map relates the two texts.
	Map EditMap
}

// Diagnostic is one compiler or lint diagnostic in the JSON form of SPEC
// Sec 16. Raw keeps the line as the compiler wrote it, for consumers that
// read the autofix candidates.
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

// parseDiagnostic decodes one line of JSON diagnostics. It reports false
// for a line that is not a diagnostic, such as the build's own messages.
func parseDiagnostic(line []byte) (Diagnostic, bool) {
	var d Diagnostic
	if err := json.Unmarshal(line, &d); err != nil || d.SchemaVersion == "" || d.File == "" {
		return Diagnostic{}, false
	}
	d.Raw = json.RawMessage(bytes.Clone(line))
	return d, true
}
