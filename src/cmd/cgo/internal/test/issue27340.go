// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

// Failed to resolve typedefs consistently.
// No runtime test; just make sure it compiles.

package cgotest

import "cmd/cgo/internal/test/issue27340"

var issue27340Var = issue27340.Issue27340GoFunc
