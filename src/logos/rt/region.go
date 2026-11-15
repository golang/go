// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import "logos/arena"

// Region is the loGo/s region handle. It is the same type as arena.Region,
// aliased so ejected programs carry regions through logos/rt.
type Region = arena.Region
