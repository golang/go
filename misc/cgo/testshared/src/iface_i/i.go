// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iface_i

type I interface {
	M()
}

type T struct {
}

func (t *T) M() {
}

// *T implements I
