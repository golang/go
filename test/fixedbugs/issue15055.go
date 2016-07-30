// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	type name string
	_ = []byte("abc", "def", 12)    // ERROR "too many arguments to conversion to \[\]byte: \(\[\]byte\)\(.abc., .def., 12\)"
	_ = string("a", "b", nil)       // ERROR "too many arguments to conversion to string: string\(.a., .b., nil\)"
	_ = []byte()                    // ERROR "missing argument to conversion to \[\]byte: \(\[\]byte\)\(\)"
	_ = string()                    // ERROR "missing argument to conversion to string: string\(\)"
	_ = name("a", 1, 3.3)           // ERROR "too many arguments to conversion to name: name\(.a., 1, 3.3\)"
	_ = map[string]string(nil, nil) // ERROR "too many arguments to conversion to map\[string\]string: \(map\[string\]string\)\(nil, nil\)"
}
