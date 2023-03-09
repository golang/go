// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package usesgenerics defines an Analyzer that checks for usage of generic
// features added in Go 1.18.
//
// # Analyzer usesgenerics
//
// usesgenerics: detect whether a package uses generics features
//
// The usesgenerics analysis reports whether a package directly or transitively
// uses certain features associated with generic programming in Go.
package usesgenerics
