// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "fmt"

type (
	_ [fmt /* ERROR invalid array length fmt */ ]int
	_ [float64 /* ERROR invalid array length float64 */ ]int
	_ [f /* ERROR invalid array length f */ ]int
	_ [nil /* ERROR invalid array length nil */ ]int
)

func f()

var _ fmt.Stringer // use fmt
