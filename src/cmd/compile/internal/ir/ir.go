// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import "cmd/compile/internal/types"

var LocalPkg *types.Pkg // package being compiled

// builtinpkg is a fake package that declares the universe block.
var BuiltinPkg *types.Pkg
