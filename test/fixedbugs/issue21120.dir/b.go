// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "reflect"

type X int

func F1() string {
	type x X

	s := struct {
		*x
	}{nil}
	v := reflect.TypeOf(s)
	return v.Field(0).PkgPath
}

func F2() string {
	type y X

	s := struct {
		*y
	}{nil}
	v := reflect.TypeOf(s)
	return v.Field(0).PkgPath
}
