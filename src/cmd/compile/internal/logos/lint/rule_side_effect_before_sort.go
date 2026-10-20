// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

// sideEffectBeforeSort is suspicious::side_effect_before_sort (SPEC Sec 1.2,
// #31): a chain lambda with a call that is not known pure before .sortBy().
// The types2 checker finds it while it checks the package
// (types2/chains_logos.go, M6), before Run is called, and the compiler hands
// the finding to Run in Input.Reported, which applies the severity of the
// suspicious group and //logos:allow like any other finding. The entry has no
// Check so the finding is reported once, and it carries the group and scope.
// The rule never applies to Go packages, which have no chain lambdas.
var sideEffectBeforeSort = Rule{
	ID:    "suspicious::side_effect_before_sort",
	Group: Suspicious,
}
