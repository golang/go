// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import (
	"fmt"
)

func f() {
	int status // ERROR syntax error: unexpected `status' at end of statement
	fmt.Println(status)
}
