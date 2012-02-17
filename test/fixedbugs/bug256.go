// errorcheck

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T U	// bogus "invalid recursive type T" from 6g
type U int

const x T = 123

type V V	// ERROR "invalid recursive type"


