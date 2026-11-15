// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

// regionEscape is correctness::region_escape (SPEC Sec 3.1, T-301.2).
var regionEscape = Rule{
	ID:    "correctness::region_escape",
	Group: Correctness,
	Go:    false,
}
