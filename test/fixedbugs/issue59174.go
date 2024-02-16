// compile

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func p() {
	s := make([]int, copy([]byte{' '}, "")-1)
	_ = append([]int{}, make([]int, len(s))...)
}
