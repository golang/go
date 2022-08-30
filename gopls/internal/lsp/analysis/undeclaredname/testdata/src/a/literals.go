// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package undeclared

type T struct{}

func literals() {
	undefinedLiterals("hey compiler", T{}, &T{}) // want "undeclared name: undefinedLiterals"
}
