// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package time

func init() {
	// force US/Pacific for time zone tests
	localOnce.Do(initTestingZone)
}

var Interrupt = interrupt
var DaysIn = daysIn
