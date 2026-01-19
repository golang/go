// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package race_test

import (
	"fmt"
	"testing"
)

func TestMain(m *testing.M) {
	m.Run()
	fmt.Println("ALL TESTS COMPLETE")
}
