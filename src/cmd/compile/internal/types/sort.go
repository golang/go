// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

// MethodsByNameCmp sorts methods by name.
func MethodsByNameCmp(x, y *Field) int {
	if x.Sym.Less(y.Sym) {
		return -1
	}
	if y.Sym.Less(x.Sym) {
		return +1
	}
	return 0
}
