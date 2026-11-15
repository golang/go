// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts

import "cmd/compile/internal/types2"

func writeAsserts(pkg *types2.Package, f *Facts) {
	scope := pkg.Scope()
	if scope == nil {
		return
	}
	for _, name := range scope.Names() {
		switch obj := scope.Lookup(name).(type) {
		case *types2.Func:
			if obj == nil || !obj.Exported() {
				continue
			}
			sig, ok := obj.Type().(*types2.Signature)
			if !ok || sig.Recv() != nil {
				continue
			}
			if tp := sig.TypeParams(); tp == nil || tp.Len() == 0 {
				continue
			}
			tpMask, rtpMask, known := obj.LogosAssertEffects()
			if !known {
				continue
			}
			fn := findOrCreateFnFact(f, name)
			fn.AssertTP = types2.LogosAssertMaskHex(tpMask)
			fn.AssertRTP = types2.LogosAssertMaskHex(rtpMask)
			fn.HasAssert = true
		case *types2.TypeName:
			if obj == nil || obj.IsAlias() {
				continue
			}
			named, ok := obj.Type().(*types2.Named)
			if !ok {
				continue
			}
			for i := range named.NumMethods() {
				m := named.Method(i)
				if m == nil {
					continue
				}
				sig, ok := m.Type().(*types2.Signature)
				if !ok {
					continue
				}
				om := m.Origin()
				if om == nil {
					continue
				}
				if sig.TypeParams().Len() == 0 && sig.RecvTypeParams().Len() == 0 {
					continue
				}
				tpMask, rtpMask, known := om.LogosAssertEffects()
				if !known {
					continue
				}
				fn := findOrCreateFnFact(f, MethodObject(name, sig.Recv(), m.Name()))
				fn.AssertTP = types2.LogosAssertMaskHex(tpMask)
				fn.AssertRTP = types2.LogosAssertMaskHex(rtpMask)
				fn.HasAssert = true
			}
		}
	}
}
