// -lang=go1.9

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// At package scope these declarations hide the predeclared names even in
// their own definitions, where they introduce declaration cycles.
type int /* ERROR "predeclared int is shadowed" */ int

type string /* ERROR "predeclared string is shadowed" */ = string

const iota /* ERROR "predeclared iota is shadowed" */ = iota

var nil /* ERROR "predeclared nil is shadowed" */ = nil
