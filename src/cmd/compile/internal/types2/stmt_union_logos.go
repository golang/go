// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
	"strings"
)

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
