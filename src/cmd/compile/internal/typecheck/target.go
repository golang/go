// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run mkbuiltin.go

package typecheck

import "cmd/compile/internal/ir"

// Target is the package being compiled.
var Target *ir.Package
