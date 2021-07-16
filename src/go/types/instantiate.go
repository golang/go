// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"fmt"
	"go/token"
)

// InstantiateLazy is like Instantiate, but avoids actually
// instantiating the type until needed.
func (check *Checker) InstantiateLazy(pos token.Pos, typ Type, targs []Type, verify bool) (res Type) {
	base := asNamed(typ)
	if base == nil {
		panic(fmt.Sprintf("%v: cannot instantiate %v", pos, typ))
	}

	return &instance{
		check:  check,
		pos:    pos,
		base:   base,
		targs:  targs,
		verify: verify,
	}
}
