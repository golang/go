// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time_test

import (
	"testing"
	. "time"
)

func TestAndroidTzdata(t *testing.T) {
	undo := ForceAndroidTzdataForTest()
	defer undo()
	if _, err := LoadLocation("America/Los_Angeles"); err != nil {
		t.Error(err)
	}
}
