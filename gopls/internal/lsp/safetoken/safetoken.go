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

// Offset returns f.Offset(pos), but first checks that the pos is in range
// for the given file.
func Offset(f *token.File, pos token.Pos) (int, error) {
	if !InRange(f, pos) {
		return -1, fmt.Errorf("pos %d is not in range [%d:%d] of file %s",
			pos, f.Base(), f.Base()+f.Size(), f.Name())
	}
	return int(pos) - f.Base(), nil
}

// Pos returns f.Pos(offset), but first checks that the offset is valid for
// the given file.
func Pos(f *token.File, offset int) (token.Pos, error) {
	if !(0 <= offset && offset <= f.Size()) {
		return token.NoPos, fmt.Errorf("offset %d is not in range for file %s of size %d", offset, f.Name(), f.Size())
	}
	return token.Pos(f.Base() + offset), nil
}

// InRange reports whether file f contains position pos.
func InRange(f *token.File, pos token.Pos) bool {
	return token.Pos(f.Base()) <= pos && pos <= token.Pos(f.Base()+f.Size())
}
