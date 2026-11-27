// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type i1 interface{ m() interface{ j1 } }
type j1 interface{ m() interface{ i1 } }

func g1[T interface{ m() interface{ j1 } }](t T) {}

func _(y j1) { g1(y) }

type i2 interface{ m() struct{ f interface{ j2 } } }
type j2 interface{ m() struct{ f interface{ i2 } } }

func g2[T interface{ m() struct{ f interface{ j2 } } }](t T) {}

func _(y j2) { g2(y) }

type i3 interface{ m() *interface{ j3 } }
type j3 interface{ m() *interface{ i3 } }

func g3[T interface{ m() *interface{ j3 } }](t T) {}

func _(y j3) { g3(y) }
