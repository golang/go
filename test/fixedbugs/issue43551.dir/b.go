// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "./a"

type S a.S
type Key a.Key

func (s S) A() Key {
	return Key(a.S(s).A())
}
