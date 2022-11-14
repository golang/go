// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build netcgo

package net

/*

// Fail if cgo isn't available.

*/
import "C"

// The build tag "netcgo" forces use of the cgo DNS resolver.
// It is the opposite of "netgo".
func init() { netCgo = true }
