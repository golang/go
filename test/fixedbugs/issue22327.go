// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Using a multi-result function as an argument to
// append should compile successfully. Previously there
// was a missing *int -> interface{} conversion that caused
// the compiler to ICE.

package p

func f() ([]interface{}, *int) {
	return nil, nil
}

var _ = append(f())
