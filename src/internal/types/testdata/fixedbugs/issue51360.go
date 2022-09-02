// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	len.Println /* ERROR cannot select on len */
	len.Println /* ERROR cannot select on len */ ()
	_ = len.Println /* ERROR cannot select on len */
	_ = len /* ERROR cannot index len */ [0]
	_ = *len /* ERROR cannot indirect len */
}
