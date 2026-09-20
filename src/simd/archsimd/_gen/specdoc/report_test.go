// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specdoc

import (
	"errors"
	"go/token"
	"strings"
	"testing"
)

// Verify that all report types implement Finding and Report implements error.
var (
	_ Finding = TypeMismatch{}
	_ Finding = NameMismatch{}
	_ Finding = DocOrderViolation{}
	_ Finding = UnknownDecl{}
	_ Finding = UnexpectedDoc{}
	_ error   = (*Report)(nil)
)

func TestReportEmpty(t *testing.T) {
	var r Report
	if !r.Empty() {
		t.Errorf("expected empty report")
	}
	if s := r.Error(); s != "" {
		t.Errorf("expected empty string, got: %q", s)
	}
}

func TestReportPositionOrder(t *testing.T) {
	r := Report{
		UnknownDecls: []UnknownDecl{
			{
				D: Decl{
					Pos:  token.Position{Filename: "b.go", Line: 20, Column: 1},
					Recv: "T",
					Name: "UnknownMethod",
				},
			},
		},
		TypeMismatches: []TypeMismatch{
			{
				D: Decl{
					Pos:  token.Position{Filename: "a.go", Line: 10, Column: 1},
					Recv: "T",
					Name: "TypeOp",
				},
				DeclSig: "func (T) TypeOp(int)",
				SpecSig: "func (T) TypeOp(int32)",
			},
		},
		DocOrderViolations: []DocOrderViolation{
			{
				D: Decl{
					Pos:  token.Position{Filename: "a.go", Line: 5, Column: 1},
					Recv: "T",
					Name: "DocOp",
				},
				Err: errors.New("comment order violation"),
			},
		},
		NameMismatches: []NameMismatch{
			{
				D: Decl{
					Pos:  token.Position{Filename: "b.go", Line: 10, Column: 1},
					Recv: "T",
					Name: "NameOp",
				},
				DeclSig: "func (T) NameOp(x int)",
				SpecSig: "func (T) NameOp(y int)",
			},
		},
		UnexpectedDocs: []UnexpectedDoc{
			{
				D: Decl{
					Pos:  token.Position{Filename: "b.go", Line: 15, Column: 1},
					Recv: "T",
					Name: "UnexpectedDocOp",
				},
			},
		},
	}

	s := r.Error()

	// Should not have any markdown section headers.
	if strings.Contains(s, "##") {
		t.Errorf("expected no section headers in report, got:\n%s", s)
	}

	// Should report in position order:
	// 1. a.go:5
	// 2. a.go:10
	// 3. b.go:10
	// 4. b.go:15
	// 5. b.go:20
	idxA5 := strings.Index(s, "a.go:5")
	idxA10 := strings.Index(s, "a.go:10")
	idxB10 := strings.Index(s, "b.go:10")
	idxB15 := strings.Index(s, "b.go:15")
	idxB20 := strings.Index(s, "b.go:20")

	if idxA5 < 0 || idxA10 < 0 || idxB10 < 0 || idxB15 < 0 || idxB20 < 0 {
		t.Fatalf("missing expected entries in report:\n%s", s)
	}

	if !(idxA5 < idxA10 && idxA10 < idxB10 && idxB10 < idxB15 && idxB15 < idxB20) {
		t.Errorf("expected positions in order a.go:5 < a.go:10 < b.go:10 < b.go:15 < b.go:20, got indices: a.go:5=%d, a.go:10=%d, b.go:10=%d, b.go:15=%d, b.go:20=%d\nReport:\n%s",
			idxA5, idxA10, idxB10, idxB15, idxB20, s)
	}
}
