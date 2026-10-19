// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
	"strings"
)

// LoGo/s M2-S3 exhaustive statement type switches on closed unions
// (SPEC Sec 2.3, T-203.1). typeSwitchStmt calls in through one hook line
// after the clause loop, once the seen map is complete.
//
// A type switch without default over a closed union operand must cover
// every member in declaration order, then nil: a missing member reports
// nonexhaustive_switch listing the misses. A default clause covers
// everything, as does a case naming a sub-union for its members (via
// logosCovers, shared with the switch-expression check). A switch on any
// other interface type has no finite member set, so only default covers
// it and nothing is reported here. The existing duplicate-case check in
// caseTypes is unaffected.
func (check *Checker) logosExhaustiveStmtSwitch(s *syntax.SwitchStmt, sx *operand, seen map[Type]syntax.Expr) {
	if !check.conf.Logos || sx == nil {
		return
	}
	tset := logosClosedUnion(sx.typ())
	if tset == nil {
		return
	}
	for _, clause := range s.Body {
		if clause != nil && clause.Cases == nil {
			return
		}
	}
	var misses []string
	for _, term := range tset.terms {
		if !check.logosCovers(term.typ, seen) {
			misses = append(misses, TypeString(term.typ, check.qualifier))
		}
	}
	if _, ok := seen[nil]; !ok {
		misses = append(misses, "nil")
	}
	if len(misses) == 0 {
		return
	}
	check.errorf(s, InvalidTypeSwitch, "%s switch on %s misses: %s", logosNonExhaust, sx.typ(), strings.Join(misses, ", "))
}
