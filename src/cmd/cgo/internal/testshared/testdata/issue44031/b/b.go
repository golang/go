// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "testshared/issue44031/a"

type T int

func (T) M() {}

var i = a.ATypeWithALoooooongName(T(0))

func F() {
	i.M()
}
