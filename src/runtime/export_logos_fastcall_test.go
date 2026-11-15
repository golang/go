// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func SetLogosFastcallOffHookForTest(fn func(fn, arg unsafe.Pointer) int32) {
	logosFastcallOffHook = fn
}

func LogosCgocallFastForTest(fn, arg unsafe.Pointer) int32 {
	return logosCgocallFast(fn, arg)
}

func LogosNocgocallbackForTest() bool {
	return getg().nocgocallback
}
