// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

// registry lists every rule, one line each. Adding a rule adds one line.
var registry = []Rule{
	infiniteRecursion,
	unsupervisedGoroutine,
	discardedError,
	exhaustiveUnionSwitch,
	arenaPointerEscape,
	concurrentMultiwordRace,
	errChainReset,
	nilDerefUnproven,
	reentrantWithLockDeadlock,
	timeAfterInLoop,
	largeBackingArrayPin,
	todoPlaceholder,
	cognitiveOverflow,
	typedNilReturn,
	deferErasesErrdeferState,
	unappliedStructDefault,
	sideEffectBeforeSort,
	serializedUnsizedInt,
	aliasExtension,
	unclosedIOCloser,
	unboundedChannelOp,
}
