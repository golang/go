// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgbits_test

import (
	"internal/pkgbits"
	"strings"
	"testing"
)

func TestRoundTrip(t *testing.T) {
	for _, version := range []pkgbits.Version{
		pkgbits.V0,
		pkgbits.V1,
		pkgbits.V2,
		pkgbits.V3,
		pkgbits.V4,
		pkgbits.V5,
	} {
		pw := pkgbits.NewPkgEncoder(version, -1)
		w := pw.NewEncoder(pkgbits.SectionMeta, pkgbits.SyncPublic)
		w.Flush()

		var b strings.Builder
		_ = pw.DumpTo(&b)
		input := b.String()

		pr := pkgbits.NewPkgDecoder("package_id", input)
		r := pr.NewDecoder(pkgbits.SectionMeta, pkgbits.PublicRootIdx, pkgbits.SyncPublic)

		if r.Version() != w.Version() {
			t.Errorf("Expected reader version %d to be the writer version %d", r.Version(), w.Version())
		}
	}
}

// Type checker to enforce that known V* have the constant values they must have.
var (
	_ [0]bool = [pkgbits.V0]bool{}
	_ [1]bool = [pkgbits.V1]bool{}
	_ [2]bool = [pkgbits.V2]bool{}
	_ [3]bool = [pkgbits.V3]bool{}
	_ [4]bool = [pkgbits.V4]bool{}
	_ [5]bool = [pkgbits.V5]bool{}
)

func TestVersions(t *testing.T) {
	type vfpair struct {
		v pkgbits.Version
		f pkgbits.Field
	}

	// has field tests
	for _, c := range []vfpair{
		{pkgbits.V1, pkgbits.Flags},
		{pkgbits.V2, pkgbits.Flags},
		{pkgbits.V0, pkgbits.HasInit},
		{pkgbits.V1, pkgbits.HasInit},
		{pkgbits.V0, pkgbits.DerivedFuncInstance},
		{pkgbits.V1, pkgbits.DerivedFuncInstance},
		{pkgbits.V0, pkgbits.DerivedInfoNeeded},
		{pkgbits.V1, pkgbits.DerivedInfoNeeded},
		{pkgbits.V2, pkgbits.AliasTypeParamNames},
		{pkgbits.V3, pkgbits.CompactCompLiterals},
		{pkgbits.V4, pkgbits.GenericMethods},
		{pkgbits.V5, pkgbits.PreserveMethodOrder},
	} {
		if !c.v.Has(c.f) {
			t.Errorf("Expected version %v to have field %v", c.v, c.f)
		}
	}

	// does not have field tests
	for _, c := range []vfpair{
		{pkgbits.V0, pkgbits.Flags},
		{pkgbits.V2, pkgbits.HasInit},
		{pkgbits.V2, pkgbits.DerivedFuncInstance},
		{pkgbits.V2, pkgbits.DerivedInfoNeeded},
		{pkgbits.V0, pkgbits.AliasTypeParamNames},
		{pkgbits.V1, pkgbits.AliasTypeParamNames},
		{pkgbits.V0, pkgbits.CompactCompLiterals},
		{pkgbits.V1, pkgbits.CompactCompLiterals},
		{pkgbits.V2, pkgbits.CompactCompLiterals},
		{pkgbits.V0, pkgbits.GenericMethods},
		{pkgbits.V1, pkgbits.GenericMethods},
		{pkgbits.V2, pkgbits.GenericMethods},
		{pkgbits.V3, pkgbits.GenericMethods},
		{pkgbits.V0, pkgbits.PreserveMethodOrder},
		{pkgbits.V1, pkgbits.PreserveMethodOrder},
		{pkgbits.V2, pkgbits.PreserveMethodOrder},
		{pkgbits.V3, pkgbits.PreserveMethodOrder},
		{pkgbits.V4, pkgbits.PreserveMethodOrder},
	} {
		if c.v.Has(c.f) {
			t.Errorf("Expected version %v to not have field %v", c.v, c.f)
		}
	}
}
