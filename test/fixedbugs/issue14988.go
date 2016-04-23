// errorcheck

// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 14988: defining a map with an invalid forward declaration array
//              key doesn't cause a fatal.

package main

type m map[k]int // ERROR "invalid map key type"
type k [1]m
