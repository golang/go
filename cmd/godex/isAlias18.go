// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.9
// +build !go1.9

package main

import "go/types"

func isAlias(obj *types.TypeName) bool {
	return false // there are no type aliases before Go 1.9
}
