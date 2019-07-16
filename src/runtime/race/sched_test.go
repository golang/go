// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race

package race_test

import (
	"bytes"
	"fmt"
	"reflect"
	"runtime"
	"testing"
)

func TestRandomScheduling(t *testing.T) {
	// Scheduler is most consistent with GOMAXPROCS=1.
	// Use that to make the test most likely to fail.
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(1))
	const N = 10
	out := make([][]int, N)
	for i := 0; i < N; i++ {
		c := make(chan int, N)
		for j := 0; j < N; j++ {
			go func(j int) {
				c <- j
			}(j)
		}
		row := make([]int, N)
		for j := 0; j < N; j++ {
			row[j] = <-c
		}
		out[i] = row
	}

	for i := 0; i < N; i++ {
		if !reflect.DeepEqual(out[0], out[i]) {
			return // found a different order
		}
	}

	var buf bytes.Buffer
	for i := 0; i < N; i++ {
		fmt.Fprintf(&buf, "%v\n", out[i])
	}
	t.Fatalf("consistent goroutine execution order:\n%v", buf.String())
}
