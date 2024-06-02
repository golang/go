// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package repro

import "testing"

func TestSomething(t *testing.T) {
	small()
	for i := 0; i < 1001; i++ {
		large(i)
	}
}
