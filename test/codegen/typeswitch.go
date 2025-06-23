// asmcheck

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

type Ix interface {
	X()
}

type Iy interface {
	Y()
}

type Iz interface {
	Z()
}

func swXYZ(a Ix) {
	switch t := a.(type) {
	case Iy: // amd64:-".*typeAssert"
		t.Y()
	case Iz: // amd64:-".*typeAssert"
		t.Z()
	}
}

type Ig[T any] interface {
	G() T
}

func swGYZ[T any](a Ig[T]) {
	switch t := a.(type) {
	case Iy: // amd64:-".*typeAssert"
		t.Y()
	case Iz: // amd64:-".*typeAssert"
		t.Z()
	case interface{ G() T }: // amd64:-".*typeAssert",-".*assertE2I\\(",".*assertE2I2"
		t.G()
	}
}

func swE2G[T any](a any) {
	switch t := a.(type) {
	case Iy:
		t.Y()
	case Ig[T]: // amd64:-".*assertE2I\\(",".*assertE2I2"
		t.G()
	}
}

func swI2G[T any](a Ix) {
	switch t := a.(type) {
	case Iy:
		t.Y()
	case Ig[T]: // amd64:-".*assertE2I\\(",".*assertE2I2"
		t.G()
	}
}

func swCaller() {
	swGYZ[int]((Ig[int])(nil))
	swE2G[int]((Ig[int])(nil))
	swI2G[int]((Ix)(nil))
}
