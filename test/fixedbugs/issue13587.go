// errorcheck -0 -l -d=wb

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test write barrier for implicit assignments to result parameters
// that have escaped to the heap.

package issue13587

import "errors"

func escape(p *error)

func F() (err error) {
	escape(&err)
	return errors.New("error") // ERROR "write barrier"
}
