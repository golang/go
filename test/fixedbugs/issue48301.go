// errorcheck

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Don't crash while reporting the error.

package p

func _() {
	type T = T // ERROR "T uses T|invalid recursive type T"
}
