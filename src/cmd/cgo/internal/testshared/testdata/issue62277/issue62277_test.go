// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue62277_test

import (
	"testing"

	"testshared/issue62277/p"
)

func TestIssue62277(t *testing.T) {
	t.Log(p.S)
	t.Log(p.T)
}
