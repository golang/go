// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mlkem

import "testing"

func TestCASTRejectionPaths(t *testing.T) {
	var rejected, notRejected bool
	testingOnlyRejectionOutcome = func(compare int) {
		if compare == 1 {
			t.Log("non-rejection path hit")
			notRejected = true
		} else {
			t.Log("rejection path hit")
			rejected = true
		}
	}
	t.Cleanup(func() {
		testingOnlyRejectionOutcome = nil
	})

	if err := fips140CAST(); err != nil {
		t.Fatal(err)
	}

	if !rejected {
		t.Error("rejection path not hit")
	}
	if !notRejected {
		t.Error("non-rejection path not hit")
	}
}
