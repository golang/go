// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

const logosConstSwitchContinue stmtContext = inTypeSwitch << 1

func (check *Checker) logosStmtEnterFunc(name string, body *syntax.BlockStmt) {
	if name == "<function literal>" {
		check.logosEnterLit(body)
	} else {
		check.logosEnterFunc(body)
	}
}

func (check *Checker) logosStmtYield(slot *syntax.Stmt) syntax.Stmt {
	s := check.logosYieldForm(*slot)
	*slot = s
	return s
}

func (check *Checker) logosStmtReadonlyCase(e syntax.Expr, T *Type) bool {
	if logosContainsReadonlyDescendants(*T) {
		check.error(e, InvalidTypeSwitch, "cannot use readonly descendants in type switch cases")
		*T = Typ[Invalid]
		return true
	}
	return false
}

func (check *Checker) logosStmtSendValue(val *operand, value syntax.Expr) {
	if !check.logosLambdaDefer(val, value) {
		check.genericExpr(val, value, nil)
	}
}

func logosStmtSwitchContext(inner stmtContext) stmtContext {
	if inner&logosConstSwitchContinue != 0 {
		inner |= continueOk
	}
	return inner
}

func (check *Checker) logosStmtSwitchOperand(x *operand) *operand {
	if x.isValid() {
		if logosContainsReadonlyDescendants(x.typ()) {
			check.error(x, InvalidTypeSwitch, "cannot use type switch on value with readonly descendants")
		} else if isTypeParam(x.typ()) {
			check.errorf(x, InvalidTypeSwitch, "cannot use type switch on type parameter value %s", x)
		} else if IsInterface(x.typ()) {
			return x
		} else {
			check.errorf(x, InvalidTypeSwitch, "%s is not an interface", x)
		}
	}
	return nil
}
