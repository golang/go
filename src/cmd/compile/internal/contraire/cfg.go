// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package contraire

// config defines the set of probabilities that Contraire will use while
// generating an AST.
type config struct {
	ConstDeclMaxCount int // [1, N]
	TypeDeclMaxCount  int // [1, N]

	// Each group below must sum to 100.
	PConstName int
	PConstLtrl int

	PTypeName int
	PTypeLtrl int

	PTypeBasic   int
	PTypePointer int
}

// cfg is a reasonable default config.
var cfg = &config{
	ConstDeclMaxCount: 4,
	TypeDeclMaxCount:  4,

	PConstName: 50,
	PConstLtrl: 50,

	PTypeName: 50,
	PTypeLtrl: 50,

	PTypeBasic:   50,
	PTypePointer: 50,
}
