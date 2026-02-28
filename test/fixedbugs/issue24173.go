// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type arrayAlias = [10]int
type mapAlias = map[int]int
type sliceAlias = []int
type structAlias = struct{}

func Exported() {
	_ = arrayAlias{}
	_ = mapAlias{}
	_ = sliceAlias{}
	_ = structAlias{}
}
