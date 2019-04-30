// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package print2

import ( // NOTE: Does not import "fmt"
	"log"
	"math/big"
)

var fmt int

func f() {
	log.Printf("%d", new(big.Int))
	log.Printf("%d", 1.0) // ERROR "Printf format %d has arg 1.0 of wrong type float64"
}
