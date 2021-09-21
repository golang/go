// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package undeclared

func selector() {
	m := map[int]bool{}
	undefinedSelector(m[1]) // want "undeclared name: undefinedSelector"
}
