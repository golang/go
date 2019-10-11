// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//+build race

// This file exists to allow us to detect that we're in race detector mode for temporarily
// disabling tests that are broken on tip with the race detector turned on.
// TODO(matloob): delete this once golang.org/issue/31749 is fixed.

package multichecker_test

func init() {
	race = true
}
