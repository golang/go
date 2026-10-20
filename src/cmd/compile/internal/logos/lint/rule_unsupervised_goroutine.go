// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

// unsupervisedGoroutine is concurrency::unsupervised_goroutine (SPEC Sec
// 1.2): a go statement in a loGo/s package outside a go group block. The
// types2 checker already reports it under this ID while it checks the
// package (M8, types2/group_logos.go), before Run is called. The entry has
// no Check so the finding is reported once, and it carries the group and
// scope so a forbid rule keeps ignoring //logos:allow and a directive that
// names it is reported as correctness::allow_forbidden_rule. It never
// applies to Go packages, where a bare go statement is ordinary Go.
var unsupervisedGoroutine = Rule{
	ID:    "concurrency::unsupervised_goroutine",
	Group: Concurrency,
}
