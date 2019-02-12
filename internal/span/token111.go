// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !go1.12

package span

import (
	"go/token"
)

// lineStart is the pre-Go 1.12 version of (*token.File).LineStart. For Go
// versions <= 1.11, we borrow logic from the analysisutil package.
// TODO(rstambler): Delete this file when we no longer support Go 1.11.
func lineStart(f *token.File, line int) token.Pos {
	// Use binary search to find the start offset of this line.

	min := 0        // inclusive
	max := f.Size() // exclusive
	for {
		offset := (min + max) / 2
		pos := f.Pos(offset)
		posn := f.Position(pos)
		if posn.Line == line {
			return pos - (token.Pos(posn.Column) - 1)
		}

		if min+1 >= max {
			return token.NoPos
		}

		if posn.Line < line {
			min = offset
		} else {
			max = offset
		}
	}
}
