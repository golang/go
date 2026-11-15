// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testlogos

/*
int stress_usleep(int micros);
#cgo noescape stress_usleep
#cgo nocallback stress_usleep
*/
import "C"

func callStressUsleep(micros int32) int32 {
	return int32(C.stress_usleep(C.int(micros)))
}
