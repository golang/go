// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts

import "cmd/compile/internal/types2"

func MethodObject(typeName string, recv *types2.Var, method string) string {
	if recv == nil {
		return typeName + "." + method
	}
	t := types2.Unalias(recv.Type())
	if cv, ok := t.(*types2.ConstView); ok {
		t = types2.Unalias(cv.Elem)
	}
	if _, ok := t.(*types2.Pointer); ok {
		return "(*" + typeName + ")." + method
	}
	return typeName + "." + method
}

func ObjectName(fn *types2.Func) string {
	sig, ok := fn.Type().(*types2.Signature)
	if !ok || sig.Recv() == nil {
		return fn.Name()
	}
	recv := types2.Unalias(sig.Recv().Type())
	isPtr := false
	if p, ok := recv.(*types2.Pointer); ok {
		isPtr = true
		recv = types2.Unalias(p.Elem())
	}
	named, ok := recv.(*types2.Named)
	if !ok || named.Obj().Name() == "" {
		return ""
	}
	if isPtr {
		return "(*" + named.Obj().Name() + ")." + fn.Name()
	}
	return named.Obj().Name() + "." + fn.Name()
}
