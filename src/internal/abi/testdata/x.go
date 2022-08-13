// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x

import "internal/abi"

func Fn0() // defined in assembly

func Fn1() {}

var FnExpr func()

func test() {
	_ = abi.FuncPCABI0(Fn0)           // line 16, no error
	_ = abi.FuncPCABIInternal(Fn0)    // line 17, error
	_ = abi.FuncPCABI0(Fn1)           // line 18, error
	_ = abi.FuncPCABIInternal(Fn1)    // line 19, no error
	_ = abi.FuncPCABI0(FnExpr)        // line 20, error
	_ = abi.FuncPCABIInternal(FnExpr) // line 21, no error
}
