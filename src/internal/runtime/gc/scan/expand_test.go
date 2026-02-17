// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package scan_test

import (
	"internal/goarch"
	"internal/runtime/gc"
	"internal/runtime/gc/scan"
	"testing"
)

type expandFunc func(sizeClass int, packed *gc.ObjMask, unpacked *gc.PtrMask)

func testExpand(t *testing.T, expF expandFunc) {
	expR := scan.ExpandReference

	testObjs(t, func(t *testing.T, sizeClass int, objs *gc.ObjMask) {
		var want, got gc.PtrMask
		expR(sizeClass, objs, &want)
		expF(sizeClass, objs, &got)

		for i := range want {
			if got[i] != want[i] {
				t.Errorf("expansion differs from reference at bit %d", i*goarch.PtrSize)
				if goarch.PtrSize == 4 {
					t.Logf("got:  %032b", got[i])
					t.Logf("want: %032b", want[i])
				} else {
					t.Logf("got:  %064b", got[i])
					t.Logf("want: %064b", want[i])
				}
			}
		}
	})
}
