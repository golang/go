// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue30768_test

import (
	"testing"

	"testshared/issue30768/issue30768lib"
)

type s struct {
	s issue30768lib.S
}

func Test30768(t *testing.T) {
	// Calling t.Log will convert S to an empty interface,
	// which will force a reference to the generated hash function,
	// defined in the shared library.
	t.Log(s{})
}
