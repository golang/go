// errchk $G -e $D/$F.go

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f(x int, y ...)	// ok

func g(x int, y float) (...)	// ERROR "[.][.][.]"

func h(x, y ...)		// ERROR "[.][.][.]"

func i(x int, y ..., z float)	// ERROR "[.][.][.]"

var x ...;		// ERROR "[.][.][.]|syntax"

type T ...;		// ERROR "[.][.][.]|syntax"
