// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check for cycles in the method call of a pointer value returned
// from a function call.

package funcptrmethcall

type T int

func (*T) pm() int {
	_ = x
	return 0
}

func pf() *T {
	return nil
}

var x = pf().pm() // ERROR "initialization cycle|depends upon itself" 
