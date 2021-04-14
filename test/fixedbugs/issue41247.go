// errorcheck

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f() [2]int {
	return [...]int{2: 0} // ERROR "cannot use \[\.\.\.\]int{...} \(type \[3\]int\)|incompatible type"
}
