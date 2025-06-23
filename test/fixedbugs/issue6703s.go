// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check for cycles in a pointer method expression.

package ptrmethexpr

type T int

func (*T) pm() int {
	_ = x
	return 0
}

var x = (*T).pm // ERROR "initialization cycle|depends upon itself"
