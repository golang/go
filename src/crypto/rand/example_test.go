// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand_test

import (
	"crypto/rand"
	"fmt"
	"math/big"
)

// ExampleInt prints a single cryptographically secure pseudorandom number between 0 and 99 inclusive.
func ExampleInt() {
	a, err := rand.Int(rand.Reader, big.NewInt(100))
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(a.Int64())
}

func ExampleRead() {
	// Note that no error handling is necessary, as Read always succeeds.
	key := make([]byte, 32)
	rand.Read(key)
}
