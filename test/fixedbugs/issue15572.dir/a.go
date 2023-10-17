// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type T struct {
}

func F() []T {
	return []T{T{}}
}

func Fi() []T {
	return []T{{}} // element with implicit composite literal type
}

func Fp() []*T {
	return []*T{&T{}}
}

func Fip() []*T {
	return []*T{{}} // element with implicit composite literal type
}

func Gp() map[int]*T {
	return map[int]*T{0: &T{}}
}

func Gip() map[int]*T {
	return map[int]*T{0: {}} // element with implicit composite literal type
}

func Hp() map[*T]int {
	return map[*T]int{&T{}: 0}
}

func Hip() map[*T]int {
	return map[*T]int{{}: 0} // key with implicit composite literal type
}
