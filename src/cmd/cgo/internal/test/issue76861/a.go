// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package issue76861

// #cgo CFLAGS: -Wall -Werror
// void issue76861(void) {}
import "C"

func Issue76861() {
	C.issue76861()
}
