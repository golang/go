// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal

import (
	"go/types"
	"slices"
)

// IsTypeNamed reports whether t is (or is an alias for) a
// package-level defined type with the given package path and one of
// the given names. It returns false if t is nil.
//
// This function avoids allocating the concatenation of "pkg.Name",
// which is important for the performance of syntax matching.
func IsTypeNamed(t types.Type, pkgPath string, names ...string) bool {
	if named, ok := types.Unalias(t).(*types.Named); ok {
		tname := named.Obj()
		return tname != nil &&
			IsPackageLevel(tname) &&
			tname.Pkg().Path() == pkgPath &&
			slices.Contains(names, tname.Name())
	}
	return false
}

// IsPointerToNamed reports whether t is (or is an alias for) a pointer to a
// package-level defined type with the given package path and one of the given
// names. It returns false if t is not a pointer type.
func IsPointerToNamed(t types.Type, pkgPath string, names ...string) bool {
	r := Unpointer(t)
	if r == t {
		return false
	}
	return IsTypeNamed(r, pkgPath, names...)
}

// IsFunctionNamed reports whether obj is a package-level function
// defined in the given package and has one of the given names.
// It returns false if obj is nil.
//
// This function avoids allocating the concatenation of "pkg.Name",
// which is important for the performance of syntax matching.
func IsFunctionNamed(obj types.Object, pkgPath string, names ...string) bool {
	f, ok := obj.(*types.Func)
	return ok &&
		IsPackageLevel(obj) &&
		f.Pkg().Path() == pkgPath &&
		f.Type().(*types.Signature).Recv() == nil &&
		slices.Contains(names, f.Name())
}

// IsMethodNamed reports whether obj is a method defined on a
// package-level type with the given package and type name, and has
// one of the given names. It returns false if obj is nil.
//
// This function avoids allocating the concatenation of "pkg.TypeName.Name",
// which is important for the performance of syntax matching.
func IsMethodNamed(obj types.Object, pkgPath string, typeName string, names ...string) bool {
	if fn, ok := obj.(*types.Func); ok {
		if recv := fn.Type().(*types.Signature).Recv(); recv != nil {
			_, T := ReceiverNamed(recv)
			return T != nil &&
				IsTypeNamed(T, pkgPath, typeName) &&
				slices.Contains(names, fn.Name())
		}
	}
	return false
}
