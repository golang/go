// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type (
	T1 interface{ comparable }
	T2 interface{ int }
)

var (
	_ comparable // ERROR cannot use type comparable outside a type constraint: interface is \(or embeds\) comparable
	_ T1         // ERROR cannot use type T1 outside a type constraint: interface is \(or embeds\) comparable
	_ T2         // ERROR cannot use type T2 outside a type constraint: interface contains type constraints
)
