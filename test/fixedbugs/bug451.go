// errorcheck

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T x.T // ERROR "undefined|expected package"

// bogus "invalid recursive type"
