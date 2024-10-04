// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package versions

import "go/build/constraint"

// ConstraintGoVersion is constraint.GoVersion (if built with go1.21+).
// Otherwise nil.
//
// Deprecate once x/tools is after go1.21.
var ConstraintGoVersion func(x constraint.Expr) string
