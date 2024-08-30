// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T *struct {}

func (x T) M () {}  // ERROR "pointer|receiver"

/*
bug046.go:7: illegal <this> pointer
*/
