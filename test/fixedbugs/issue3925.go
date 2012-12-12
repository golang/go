// errorcheck

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 3925: wrong line number for error message "missing key in map literal"

// also a test for correct line number in other malformed composite literals.

package foo

var _ = map[string]string{
	"1": "2",
	"3", "4", // ERROR "missing key"
}

var _ = []string{
	"foo",
	"bar",
	20, // ERROR "cannot use"
}

