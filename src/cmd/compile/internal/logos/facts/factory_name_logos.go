// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts

import (
	"sort"
	"strconv"
	"strings"

	"cmd/compile/internal/types2"
)

// DefaultFactoryName is the generated declaration-scope binding for one
// defaulted field. It is exported so ejected and cross-package consumers can
// reference it; declaration scope keeps private names and interface dynamic
// types intact while every call allocates fresh composite storage. The double
// underscore keeps (type, field) pairs like A_B.C and A.B_C distinct.
func DefaultFactoryName(typeName, field string) string {
	return "LogosDefault_" + typeName + "__" + field
}

// FactoryNamePlan resolves the emitted symbol for every owner.field default
// pair in the package. A generated name that collides with another generated
// name or with an existing package-scope declaration gets a deterministic
// "__N" suffix, so the factory always exists under the name that call sites
// and exported fact rows reference. Owner-less and "_" fields are omitted.
// Keys are sorted for determinism: the exported fact for a renamed factory
// must name the same symbol the package emits.
func FactoryNamePlan(info *types2.Info, tpkg *types2.Package) map[string]string {
	plan := map[string]string{}
	if info == nil {
		return plan
	}
	var keys []string
	seen := map[string]bool{}
	for _, defs := range info.LogosFieldDefaults {
		for _, d := range defs {
			key := d.Owner + "." + d.Field
			if d.Owner == "" || d.Field == "_" || seen[key] {
				continue
			}
			seen[key] = true
			keys = append(keys, key)
		}
	}
	sort.Strings(keys)
	used := map[string]bool{}
	free := func(n string) bool {
		if used[n] {
			return false
		}
		if tpkg != nil && tpkg.Scope() != nil && tpkg.Scope().Lookup(n) != nil {
			return false
		}
		return true
	}
	for _, key := range keys {
		i := strings.IndexByte(key, '.')
		base := DefaultFactoryName(key[:i], key[i+1:])
		name := base
		for n := 2; !free(name); n++ {
			name = base + "__" + strconv.Itoa(n)
		}
		used[name] = true
		plan[key] = name
	}
	return plan
}
