// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package safetoken provides wrappers around methods in go/token, that return
// errors rather than panicking.
package safetoken

import (
	"fmt"
	"go/token"
)

// Offset returns tok.Offset(pos), but first checks that the pos is in range
// for the given file.
func Offset(tok *token.File, pos token.Pos) (int, error) {
	if !InRange(tok, pos) {
		return -1, fmt.Errorf("pos %v is not in range for file [%v:%v)", pos, tok.Base(), tok.Base()+tok.Size())
	}
	return tok.Offset(pos), nil
}

// Pos returns tok.Pos(offset), but first checks that the offset is valid for
// the given file.
func Pos(tok *token.File, offset int) (token.Pos, error) {
	if offset < 0 || offset > tok.Size() {
		return token.NoPos, fmt.Errorf("offset %v is not in range for file of size %v", offset, tok.Size())
	}
	return tok.Pos(offset), nil
}

// InRange reports whether the given position is in the given token.File.
func InRange(tok *token.File, pos token.Pos) bool {
	size := tok.Pos(tok.Size())
	return int(pos) >= tok.Base() && pos <= size
}
