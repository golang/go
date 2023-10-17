// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package plugin_test

import (
	_ "plugin"
	"testing"
)

func TestPlugin(t *testing.T) {
	// This test makes sure that executable that imports plugin
	// package can actually run. See issue #28789 for details.
}
