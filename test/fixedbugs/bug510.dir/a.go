// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import "reflect"

type A = map[int] bool

func F() interface{} {
	return reflect.New(reflect.TypeOf((*A)(nil))).Elem().Interface()
}
