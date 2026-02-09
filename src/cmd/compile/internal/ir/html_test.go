// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestHTMLWriter(t *testing.T) {
	// Initialize base.Ctxt to avoid panics
	base.Ctxt = new(obj.Link)

	// Setup a temporary directory for output
	tmpDir := t.TempDir()

	// Mock func
	fn := &Func{
		Nname: &Name{
			sym: &types.Sym{Name: "TestFunc"},
		},
	}
	// Func embeds miniExpr, so we might need to set op if checked
	fn.op = ODCLFUNC

	// Create HTMLWriter
	outFile := filepath.Join(tmpDir, "test.html")
	w := NewHTMLWriter(outFile, fn, "")
	if w == nil {
		t.Fatalf("Failed to create HTMLWriter")
	}

	// Write a phase
	w.WritePhase("phase1", "Phase 1")

	// Register a file/line
	posBase := src.NewFileBase("test.go", "test.go")
	// base.Ctxt.PosTable.Register(posBase) -- Not needed/doesn't exist
	pos := src.MakePos(posBase, 10, 1)

	// Create a dummy node
	n := &Name{
		sym:   &types.Sym{Name: "VarX"},
		Class: PAUTO,
	}
	n.op = ONAME
	n.pos = base.Ctxt.PosTable.XPos(pos)

	// Add another phase which actually dumps something interesting
	fn.Body = []Node{n}
	w.WritePhase("phase2", "Phase 2")

	// Test escaping
	n2 := &Name{
		sym:   &types.Sym{Name: "<Bad>"},
		Class: PAUTO,
	}
	n2.op = ONAME
	fn.Body = []Node{n2}
	w.WritePhase("phase3", "Phase 3")

	w.Close()

	// Verify file exists and has content
	content, err := os.ReadFile(outFile)
	if err != nil {
		t.Fatalf("Failed to read output file: %v", err)
	}

	s := string(content)
	if len(s) == 0 {
		t.Errorf("Output file is empty")
	}

	// Check for Expected strings
	expected := []string{
		"<html>",
		"Phase 1",
		"Phase 2",
		"Phase 2",
		"VarX",
		"NAME",
		"&lt;Bad&gt;",
		"resizer",
		"loc-",
		"line-number",
		"sym-",
		"variable-name",
	}

	for _, e := range expected {
		if !strings.Contains(s, e) {
			t.Errorf("Output missing %q", e)
		}
	}
}
