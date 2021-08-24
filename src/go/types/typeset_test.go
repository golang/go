// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import "testing"

func TestInvalidTypeSet(t *testing.T) {
	if !invalidTypeSet.IsEmpty() {
		t.Error("invalidTypeSet is not empty")
	}
}

// TODO(gri) add more tests
