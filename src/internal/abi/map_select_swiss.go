// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.swissmap

package abi

// Select the map type that this binary is built using. This is for common
// lookup methods like Type.Key to know which type to use.
//
// Note that mapType *must not be used by any functions called in the
// compiler to build a target program* because the compiler must use the map
// type determined by run-time GOEXPERIMENT, not the build tags used to build
// the compiler.
//
// TODO(prattmic): This package is rather confusing because it has many
// functions that can't be used by the compiler (e.g., Type.Uncommon depends on
// the layout of type + uncommon objects in the binary. It would be incorrect
// for an ad-hoc local Type object). It may be best to move code that isn't
// usable by the compiler out of the package.
type mapType = SwissMapType
