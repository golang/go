// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

// nilDerefUnproven is correctness::nil_deref_unproven (SPEC Sec 1.2, #99,
// T-202): a field or method selection on an option, before a nil test proves
// the option present. The M3 checker enforces it under this ID while it checks
// the package (types2/option_logos.go, with the narrowing of narrow_logos.go),
// before Run is called, and the finding stops the compile there. The entry
// has no Check so the finding is reported once, and it carries the group and
// scope so a forbid rule keeps ignoring //logos:allow and a directive that
// names it is reported as correctness::allow_forbidden_rule whenever the
// package reaches the lint pass. The compiler routes the checker's error
// through NilDerefResult, which adds the JSON autofix candidates of SPEC Sec
// 1.2: optional chaining with a ?? default, and an early-return guard. The
// rule never applies to Go packages, which have no options.
var nilDerefUnproven = Rule{
	ID:    nilDerefID,
	Group: Correctness,
}
