// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

var GS string

func M() string {
	if s := getname("Fred"); s != "" {
		return s
	}
	if s := getname("Joe"); s != "" {
		return s
	}

	return string("Alex")
}

// getname can be any function returning a string, just has to be non-inlinable.

//go:noinline
func getname(s string) string {
	return s + "foo"
}
