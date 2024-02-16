// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is tested when running "go test -run Manual"
// without source arguments. Use for one-off debugging.

package p

type T[P any, B *P] struct{}

func (T /* ERROR "cannot use generic type" */) m0() {}

// TODO(rfindley): eliminate the duplicate errors here.
func ( /* ERROR "got 1 type parameter, but receiver base type declares 2" */ T /* ERROR "not enough type arguments for type" */ [_]) m1() {
}
func (T[_, _]) m2() {}

// TODO(gri) this error is unfortunate (issue #51343)
func (T /* ERROR "too many type arguments for type" */ [_, _, _]) m3() {}
