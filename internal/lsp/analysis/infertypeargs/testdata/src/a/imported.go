// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import "a/imported"

func _() {
	var x int
	imported.F[int](x) // want "unnecessary type arguments"
}
