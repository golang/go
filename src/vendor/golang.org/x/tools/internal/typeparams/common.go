// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package typeparams contains common utilities for writing tools that
// interact with generic Go code, as introduced with Go 1.18. It
// supplements the standard library APIs. Notably, [TypeSetOf]
// computes a minimal representation of the type set of a type
// parameter.
//
// TODO(adonovan): This package should be melted down or moved
// elsewhere (such as go/types, x/tools/go/types/typeutil, or
// x/tools/internal/typesinternal). Some of these APIs are available
// externally in the golang.org/x/exp/typeparams module, though the
// implementations may be stale.
package typeparams

import "go/types"

// IsTypeParam reports whether t is a type parameter (or an alias of one).
func IsTypeParam(t types.Type) bool {
	_, ok := types.Unalias(t).(*types.TypeParam)
	return ok
}
