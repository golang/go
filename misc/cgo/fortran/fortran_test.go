// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fortran

import "testing"

func TestFortran(t *testing.T) {
	if a := TheAnswer(); a != 42 {
		t.Errorf("Unexpected result for The Answer. Got: %d Want: 42", a)
	}
}
