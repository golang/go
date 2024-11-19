// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package synctest provides support for testing concurrent code.
//
// See the testing/synctest package for function documentation.
package synctest

import (
	_ "unsafe" // for go:linkname
)

//go:linkname Run
func Run(f func())

//go:linkname Wait
func Wait()
