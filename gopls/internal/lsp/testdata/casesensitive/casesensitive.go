// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package casesensitive

func _() {
	var lower int //@item(lower, "lower", "int", "var")
	var Upper int //@item(upper, "Upper", "int", "var")

	l //@casesensitive(" //", lower)
	U //@casesensitive(" //", upper)

	L //@casesensitive(" //")
	u //@casesensitive(" //")
}
