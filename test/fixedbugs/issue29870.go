// compile

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure we can compile "_" functions without crashing.

package main

import "log"

func _() {
	log.Println("%2F")
}
