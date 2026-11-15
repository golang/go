// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import . "internal/types/errors"

func (check *Checker) logosDeclScheduleVarExports(obj *Var, lhs []*Var) {
	if obj.Parent() == check.pkg.scope {
		check.later(func() {
			if obj.Exported() {
				check.logosDeclReadonlyExport(obj)
			}
			for _, v := range lhs {
				if v != obj && v.Exported() {
					check.logosDeclReadonlyExport(v)
				}
			}
		}).describef(obj, "readonly variable exports")
	}
}

func (check *Checker) logosDeclScheduleTypeExport(obj *TypeName) {
	if obj.Exported() && obj.Parent() == check.pkg.scope {
		check.later(func() {
			check.logosDeclReadonlyExport(obj)
		}).describef(obj, "readonly export %s", obj.name)
	}
}

func (check *Checker) logosDeclScheduleFuncExport(obj *Func) {
	if obj.Exported() {
		check.later(func() {
			check.logosDeclReadonlyExport(obj)
		}).describef(obj, "readonly export %s", obj.name)
	}
}

func (check *Checker) logosDeclReadonlyExport(obj Object) {
	if logosContainsReadonlyDescendants(obj.Type()) {
		check.errorf(obj, BadDecl, "cannot export %s with readonly descendants before qualification export encoding is supported", obj.Name())
	}
}
