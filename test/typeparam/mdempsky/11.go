// errorcheck

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reported by Cuong Manh Le.

package p

type a struct{}

//go:notinheap
type b a

var _ = (*b)(new(a)) // ERROR "cannot convert"
