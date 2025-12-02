// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.p

package refactor

// This is the only file in this package that should import analysis.
//
// TODO(adonovan): consider unaliasing the type to break the
// dependency. (The ergonomics of slice append are unfortunate.)

import "golang.org/x/tools/go/analysis"

// An Edit describes a deletion and/or an insertion.
type Edit = analysis.TextEdit
