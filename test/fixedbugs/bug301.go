// compile

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// https://golang.org/issue/990

package main

func main() {
	defer func() {
		if recover() != nil {
			panic("non-nil recover")
		}
	}()
	panic(nil)
}
