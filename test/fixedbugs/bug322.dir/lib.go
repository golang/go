// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lib

type T struct {
	x int  // non-exported field
}

func (t T) M() {
}

func (t *T) PM() {
}
