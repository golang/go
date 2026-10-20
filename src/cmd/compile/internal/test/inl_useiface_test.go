// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"internal/testenv"
	"path/filepath"
	"testing"
)

// TestInlineBudgetKeepsHashfuncOutOfUseIface links the hash/maphash test
// binary with raised inline budgets.
//
// At these budgets the closures in TestComparableAllocations inline
// maphash.Comparable together with comparableHash, so the SSA fixed-load
// rewrite folds the map type's Hasher field to the generated
// type:.hashfunc.* closure. That symbol is not a runtime type, and marking
// it R_USEIFACE makes the linker panic:
//
//	R_USEIFACE in hash/maphash.TestComparableAllocations.func2 references
//	type:.hashfunc.M8S which is not a type or itab
//
// The two budgets fold different hash signatures (M8S and SS).
func TestInlineBudgetKeepsHashfuncOutOfUseIface(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	t.Parallel()

	for _, budget := range []string{"120", "400"} {
		t.Run("inlbudget="+budget, func(t *testing.T) {
			exe := filepath.Join(t.TempDir(), "maphash.test.exe")
			cmd := testenv.Command(t, testenv.GoToolPath(t), "test", "-c", "-o", exe,
				"-gcflags=hash/maphash=-d=inlbudget="+budget, "hash/maphash")
			out, err := cmd.CombinedOutput()
			if err != nil {
				t.Fatalf("linking hash/maphash.test at inlbudget=%s failed: %v\n%s", budget, err, out)
			}
		})
	}
}
