// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file checks invariants of token.Token ordering that we rely on
// since package go/token doesn't provide any guarantees at the moment.

package types

import (
	"go/token"
	"testing"
)

var assignOps = map[token.Token]token.Token{
	token.ADD_ASSIGN:     token.ADD,
	token.SUB_ASSIGN:     token.SUB,
	token.MUL_ASSIGN:     token.MUL,
	token.QUO_ASSIGN:     token.QUO,
	token.REM_ASSIGN:     token.REM,
	token.AND_ASSIGN:     token.AND,
	token.OR_ASSIGN:      token.OR,
	token.XOR_ASSIGN:     token.XOR,
	token.SHL_ASSIGN:     token.SHL,
	token.SHR_ASSIGN:     token.SHR,
	token.AND_NOT_ASSIGN: token.AND_NOT,
}

func TestZeroTok(t *testing.T) {
	// zero value for token.Token must be token.ILLEGAL
	var zero token.Token
	if token.ILLEGAL != zero {
		t.Errorf("%s == %d; want 0", token.ILLEGAL, zero)
	}
}

func TestAssignOp(t *testing.T) {
	// there are fewer than 256 tokens
	for i := 0; i < 256; i++ {
		tok := token.Token(i)
		got := assignOp(tok)
		want := assignOps[tok]
		if got != want {
			t.Errorf("for assignOp(%s): got %s; want %s", tok, got, want)
		}
	}
}
