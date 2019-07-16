// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package go1

// benchmark based on time/time_test.go

import (
	"testing"
	"time"
)

func BenchmarkTimeParse(b *testing.B) {
	for i := 0; i < b.N; i++ {
		time.Parse(time.ANSIC, "Mon Jan  2 15:04:05 2006")
	}
}

func BenchmarkTimeFormat(b *testing.B) {
	t := time.Unix(1265346057, 0)
	for i := 0; i < b.N; i++ {
		t.Format("Mon Jan  2 15:04:05 2006")
	}
}
