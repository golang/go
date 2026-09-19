// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package defers

import "time"

func _() {
	defer time.Since(time.Now()) // ERROR "call to time.Since is not deferred"
}
