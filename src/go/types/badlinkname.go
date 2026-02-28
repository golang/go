// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import _ "unsafe"

// This should properly be in infer.go, but that file is auto-generated.

// infer should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/goplus/gox
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname badlinkname_Checker_infer go/types.(*Checker).infer
func badlinkname_Checker_infer(*Checker, positioner, []*TypeParam, []Type, *Tuple, []*operand, bool, *error_) []Type
