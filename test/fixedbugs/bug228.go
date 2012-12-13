// errorcheck

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f(x int, y ...int)	// ok

func g(x int, y float32) (...)	// ERROR "[.][.][.]" "final argument"

func h(x, y ...int)		// ERROR "[.][.][.]"

func i(x int, y ...int, z float32)	// ERROR "[.][.][.]"

var x ...int;		// ERROR "[.][.][.]|syntax|type"

type T ...int;		// ERROR "[.][.][.]|syntax|type"
