// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type S1 struct{}
type S2 struct{}

func _[P *S1|*S2]() {
	_= []P{{ /* ERROR invalid composite literal element type P: no core type */ }}
}

func _[P *S1|S1]() {
	_= []P{{ /* ERROR invalid composite literal element type P: no core type */ }}
}
