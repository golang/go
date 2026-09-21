// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package timeformat

import "time"

func _(t time.Time) {
	_ = t.Format("2006-02-01") // ERROR "2006-02-01 should be 2006-01-02"
}
