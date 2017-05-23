// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fortran

// int the_answer();
import "C"

func TheAnswer() int {
	return int(C.the_answer())
}
