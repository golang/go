// compile

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 17918: slice out-of-bounds in ssa/cse

package dead

import (
	"fmt"
	"time"
)

var (
	units = []struct {
		divisor time.Duration
		unit    rune
	}{
		{1000000, 's'},
		{60, 'm'},
		{60, 'h'},
		{24, 'd'},
		{7, 'w'},
	}
)

func foobar(d time.Duration) string {
	d /= time.Microsecond
	unit := 'u'

	for _, f := range units {
		if d%f.divisor != 0 {
			break
		}
		d /= f.divisor
		unit = f.unit
	}
	return fmt.Sprintf("%d%c", d, unit)
}
