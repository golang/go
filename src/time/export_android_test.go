// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

func ForceAndroidTzdataForTest() (undo func()) {
	allowGorootSource = false
	origLoadFromEmbeddedTZData := loadFromEmbeddedTZData
	loadFromEmbeddedTZData = nil

	return func() {
		allowGorootSource = true
		loadFromEmbeddedTZData = origLoadFromEmbeddedTZData
	}
}
