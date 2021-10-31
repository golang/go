// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package typeparams

import "fmt"

type N[T any] int

func (N[P]) Wrapf(p P, format string, args ...interface{}) { // want Wrapf:"printfWrapper"
	fmt.Printf(format, args...)
}

func (*N[P]) PtrWrapf(p P, format string, args ...interface{}) { // want PtrWrapf:"printfWrapper"
	fmt.Printf(format, args...)
}

func Printf[P any](p P, format string, args ...interface{}) { // want Printf:"printfWrapper"
	fmt.Printf(format, args...)
}
