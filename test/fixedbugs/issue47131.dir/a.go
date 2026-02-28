// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type MyInt int

type MyIntAlias = MyInt

func (mia *MyIntAlias) Get() int {
	return int(*mia)
}
