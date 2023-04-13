// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Set the defaultResolver to resolverCgo when the netcgo build tag is being used.

//go:build netcgo

package net

/*

// Fail if cgo isn't available.

*/
import "C"

func init() { defaultResolver = resolverCgo }
