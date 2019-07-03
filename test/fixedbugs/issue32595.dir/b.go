// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "reflect"

func B() {
	t1 := reflect.TypeOf([0]byte{})
	t2 := reflect.TypeOf(new([0]byte)).Elem()
	if t1 != t2 {
		panic("[0]byte types do not match")
	}
}
