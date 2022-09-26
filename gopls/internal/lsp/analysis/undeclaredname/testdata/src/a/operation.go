// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package undeclared

import "time"

func operation() {
	undefinedOperation(10 * time.Second) // want "(undeclared name|undefined): undefinedOperation"
}
