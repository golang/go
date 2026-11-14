// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

func (check *Checker) logosCallReadonlyConversion(x *operand, T Type) bool {
	return check.conf.Logos && check.logosReadonlyBoundary(x, T, "conversion")
}

func (check *Checker) logosCallReadonlyArguments(sigParams *Tuple, args []*operand) {
	if check.conf.Logos {
		for i, a := range args {
			check.logosReadonlyBoundary(a, sigParams.vars[i].typ, "argument")
		}
	}
}

func (check *Checker) logosCallResolveLambdas(sigParams *Tuple, args []*operand) {
	for i, a := range args {
		check.logosLambdaResolve(a, sigParams.vars[i].typ)
	}
}
