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
			t.Errorf("Expected reader version %q to be the writer version %q", r.Version(), w.Version())
		}
	}
}

// Type checker to enforce that know V* have the constant values they must have.
var _ [0]bool = [pkgbits.V0]bool{}
var _ [1]bool = [pkgbits.V1]bool{}

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
	} {
		if c.v.Has(c.f) {
			t.Errorf("Expected version %v to not have field %v", c.v, c.f)
		}
	}
}
