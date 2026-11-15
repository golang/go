// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

// guardedAccess is concurrency::unguarded_access (SPEC Sec 1.2, #195): a read
// or write of a field marked //logos:guardedby with no holding of its guard
// live at the access. The analysis is flow sensitive over the statement
// structure: branches merge by intersection, a deferred unlock keeps the
// lock held to function exit, a deferred function literal runs in LIFO order
// against each exit path's held set, and a call applies its callee's
// declares effects. Internal-pointer-before escape special case here lives
// in lockAudit. The rule is language neutral: Go packages take it on too.
var guardedAccess = Rule{
	ID:    guardedByRule,
	Group: Concurrency,
	Go:    true,
	Check: func(pass *Pass) {
		for _, d := range analyzeLocks(pass).unguarded {
			pass.Report(d)
		}
	},
}

// nonatomicAccess is concurrency::nonatomic_access (SPEC Sec 1.2): a read,
// write, or leaking address of a field marked //logos:atomic outside a
// sync/atomic call. Language neutral.
var nonatomicAccess = Rule{
	ID:    atomicRule,
	Group: Concurrency,
	Go:    true,
	Check: func(pass *Pass) {
		for _, d := range analyzeLocks(pass).nonatomic {
			pass.Report(d)
		}
	},
}

// lockRequirement is concurrency::lock_requirement_unmet (SPEC Sec 1.2): a
// call of a function marked //logos:requires reports when the named lock is
// not held at the call; the contract's acquires and releases move the held
// set across the call. Language neutral.
var lockRequirement = Rule{
	ID:    lockRequireRule,
	Group: Concurrency,
	Go:    true,
	Check: func(pass *Pass) {
		for _, d := range analyzeLocks(pass).unmet {
			pass.Report(d)
		}
	},
}
