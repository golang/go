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
func Offset(tf *token.File, pos token.Pos) (int, error) {
	if !InRange(tf, pos) {
		return -1, fmt.Errorf("pos %v is not in range for file [%v:%v)", pos, tf.Base(), tf.Base()+tf.Size())
	}
	return tf.Offset(pos), nil
}

// Pos returns tok.Pos(offset), but first checks that the offset is valid for
// the given file.
func Pos(tf *token.File, offset int) (token.Pos, error) {
	if offset < 0 || offset > tf.Size() {
		return token.NoPos, fmt.Errorf("offset %v is not in range for file of size %v", offset, tf.Size())
	}
	return tf.Pos(offset), nil
}

// InRange reports whether the given position is in the given token.File.
func InRange(tf *token.File, pos token.Pos) bool {
	size := tf.Pos(tf.Size())
	return int(pos) >= tf.Base() && pos <= size
}
