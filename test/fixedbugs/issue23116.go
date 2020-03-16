// errorcheck

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f(x interface{}) {
	switch x.(type) {
	}

	switch t := x.(type) { // ERROR "declared but not used"
	}
}
