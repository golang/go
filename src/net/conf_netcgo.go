// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Set the defaultResolver to resovlerCgo when the netcgo build tag is being used,
// but not when the defaultResolver is being set by conf_netgo.go, so that netgo
// always takes precendence over netcgo.

//go:build netcgo && !(netgo || (!cgo && !darwin && !windows))

package net

/*

// Fail if cgo isn't available.

*/
import "C"

func init() { defaultResolver = resolverCgo }
