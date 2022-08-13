// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	len. /* ERROR cannot select on len */ Println
	len. /* ERROR cannot select on len */ Println()
	_ = len. /* ERROR cannot select on len */ Println
	_ = len[ /* ERROR cannot index len */ 0]
	_ = *len /* ERROR cannot indirect len */
}
