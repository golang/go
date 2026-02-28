// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _[P1 any, P2 ~byte, P3 []int | []byte](s1 P1, s2 P2, s3 P3) {
	_ = append(nil /* ERROR "invalid append: argument must be a slice; have untyped nil" */, 0)
	_ = append(s1 /* ERROR "invalid append: argument must be a slice; have s1 (variable of type P1 constrained by any)" */, 0)
	_ = append(s2 /* ERROR "invalid append: argument must be a slice; have s2 (variable of type P2 constrained by ~byte)" */, 0)
	_ = append(s3 /* ERROR "invalid append: mismatched slice element types int and byte in s3 (variable of type P3 constrained by []int | []byte)" */, 0)
}
