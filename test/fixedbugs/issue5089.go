// errorcheck

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 5089: gc allows methods on non-locals if symbol already exists

package p

import "bufio"	// GCCGO_ERROR "previous"

func (b *bufio.Reader) Buffered() int { // ERROR "non-local|redefinition"
	return -1
}
