// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"io"
	"slices"
	"strings"
	"testing"
	"testing/iotest"
)

const logosConstPeekSrc = `package p

func f() {
	const/*c*/if 1 == 1 {
	}
	const /* a ** / b **/ switch int {
	}
	const // line
	if 1 == 1 {
	}
	const inform = 3
	const ifé = 1
	const x = 1
	const (
		y = 2
	)
	const /* open
}
`

var logosConstPeekWant = []string{
	"ConstIfStmt",
	"ConstSwitchStmt",
	"ConstIfStmt",
	"DeclStmt",
	"DeclStmt",
	"DeclStmt",
	"DeclStmt",
}

func TestLogosConstPeek(t *testing.T) {
	readers := map[string]func(string) io.Reader{
		"whole":    func(s string) io.Reader { return strings.NewReader(s) },
		"one-byte": func(s string) io.Reader { return iotest.OneByteReader(strings.NewReader(s)) },
	}
	for name, mk := range readers {
		t.Run(name, func(t *testing.T) {
			var errs []string
			errh := func(err error) { errs = append(errs, err.Error()) }
			file, _ := Parse(NewFileBase("peek.lgo"), mk(logosConstPeekSrc), errh, nil, LogosMode)
			var got []string
			for _, s := range file.DeclList[0].(*FuncDecl).Body.List {
				got = append(got, logosStmtKind(s))
			}
			if len(got) < len(logosConstPeekWant) || !slices.Equal(got[:len(logosConstPeekWant)], logosConstPeekWant) {
				t.Errorf("statements = %v, want prefix %v", got, logosConstPeekWant)
			}
			if len(errs) == 0 {
				t.Errorf("want an error for the unterminated comment")
			}
		})
	}
}

func logosStmtKind(s Stmt) string {
	switch s.(type) {
	case *ConstIfStmt:
		return "ConstIfStmt"
	case *ConstSwitchStmt:
		return "ConstSwitchStmt"
	case *DeclStmt:
		return "DeclStmt"
	}
	return "other"
}
