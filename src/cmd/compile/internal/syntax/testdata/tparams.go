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
	m /* ERROR method must have no type parameters */ [_ _, _ /* ERROR missing parameter type */ ]()
	t[a, b]
}

func f[ /* ERROR empty type parameter list */ ]()
func f[a, b /* ERROR missing type constraint */ ]()
func f[a t, b t, c /* ERROR missing type constraint */ ]()

func f[a b,  /* ERROR expected ] */ 0] ()

// go.dev/issue/49482
type (
	t[a *[]int] struct{}
	t[a *t,] struct{}
	t[a *t|[]int] struct{}
	t[a *t|t,] struct{}
	t[a *t|~t,] struct{}
	t[a *struct{}|t] struct{}
	t[a *t|struct{}] struct{}
	t[a *struct{}|~t] struct{}
)

// go.dev/issue/51488
type (
	t[a *t|t,] struct{}
	t[a *t|t, b t] struct{}
	t[a *t|t] struct{}
	t[a *[]t|t] struct{}
	t[a ([]t)] struct{}
	t[a ([]t)|t] struct{}
)

// go.dev/issue/60812
type (
	t [t]struct{}
	t [[]t]struct{}
	t [[t]t]struct{}
	t [/* ERROR missing type parameter name or invalid array length */ t[t]]struct{}
	t [t t[t], /* ERROR missing type parameter name */ t[t]]struct{}
	t [/* ERROR missing type parameter name */ t[t], t t[t]]struct{}
	t [/* ERROR missing type parameter name */ t[t], t[t]]struct{} // report only first error
)
