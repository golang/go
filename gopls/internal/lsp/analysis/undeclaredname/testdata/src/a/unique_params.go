// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package undeclared

func uniqueArguments() {
	var s string
	var i int
	undefinedUniqueArguments(s, i, s) // want "undeclared name: undefinedUniqueArguments"
}
