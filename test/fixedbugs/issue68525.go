// run -gcflags='all=-N -l'

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "os"

func main() {
	os.OpenFile(os.DevNull, os.O_WRONLY, 0)
}
