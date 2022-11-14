// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _(x int, c string) {
	switch x {
	case c /* ERROR invalid case c in switch on x \(mismatched types string and int\) */ :
	}
}

func _(x, c []int) {
	switch x {
	case c /* ERROR invalid case c in switch on x \(slice can only be compared to nil\) */ :
	}
}
