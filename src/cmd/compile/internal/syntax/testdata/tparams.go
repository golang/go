// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type t[a, b /* ERROR missing type constraint */ ] struct{}
type t[a t, b t, c /* ERROR missing type constraint */ ] struct{}
type t struct {
	t [n]byte
	t[a]
	t[a, b]
}
type t interface {
	t[a]
	m /* ERROR method must have no type parameters */ [_ _, /* ERROR mixed */ _]()
	t[a, b]
}

func f[ /* ERROR empty type parameter list */ ]()
func f[a, b /* ERROR missing type constraint */ ]()
func f[a t, b t, c /* ERROR missing type constraint */ ]()

func f[a b,  /* ERROR expecting ] */ 0] ()
