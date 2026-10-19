// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s scanner logic for M2-S5 (`=>`). scanner.go
// calls into it through one hook line only (G3).

// logosFatArrow completes the `=>` token. It is called right after the
// scanner consumed '=' and found no second '='. In LogosMode, a '>'
// that joins the '=' with no gap makes the pair one _FatArrow token
// (SPEC Sec 5.1, T-104.6); the token ends no operand, so a following
// newline yields no ';'. Otherwise it reports false and leaves the
// scanner untouched, so plain Go keeps scanning `=` followed by `>`
// exactly as upstream and `= >` with a gap keeps both tokens in
// LogosMode.
func (s *scanner) logosFatArrow() bool {
	if s.mode&uint(LogosMode) == 0 || s.ch != '>' {
		return false
	}
	s.nextch()
	s.tok = _FatArrow
	return true
}
