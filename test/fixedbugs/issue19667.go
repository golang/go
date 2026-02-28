// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure we don't crash when reporting this error.

package p

func f() {
	if err := http.ListenAndServe( // GCCGO_ERROR "undefined name"
} // ERROR "unexpected }, expected expression|expected operand|missing .*\)|expected .*;|expected .*{"
