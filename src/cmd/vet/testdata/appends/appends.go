// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the appends checker.

package appends

func AppendsTest() {
	sli := []string{"a", "b", "c"}
	sli = append(sli) // ERROR "append with no values"
}
