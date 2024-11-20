// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand_test

import "crypto/rand"

func ExampleRead() {
	// Note that no error handling is necessary, as Read always succeeds.
	key := make([]byte, 32)
	rand.Read(key)
}
