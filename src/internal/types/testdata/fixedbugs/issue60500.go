// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _() {
	log("This is a test %v" /* ERROR "cannot use \"This is a test %v\" (untyped string constant) as bool value in argument to log" */, "foo")
}

func log(enabled bool, format string, args ...any)
