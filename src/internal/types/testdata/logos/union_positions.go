// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// Plain-Go twin of union_members.lgo: the same declarations and value
// positions with no .lgo-only syntax. Union interfaces are legal as
// declared types, but standard Go rejects every value position with its
// own diagnostic, proving .go is unchanged where .lgo accepts closed
// unions and reports logos tags.

type U interface {
	int | string
}

var u U /* ERROR "interface contains type constraints" */

type Open interface {
	~int | ~string
}

var _ Open // ERROR "interface contains type constraints"

var _ interface /* ERROR "interface contains type constraints" */ {
	int | string
}

var v interface /* ERROR "interface contains type constraints" */ {
	~int | ~string
}
