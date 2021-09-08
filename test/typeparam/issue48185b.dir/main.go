// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"a"
	"fmt"
)

func main() {
	_ = a.UnmarshalOptions1{
		Unmarshalers: a.UnmarshalFuncV2(func(opts a.UnmarshalOptions1, dec *a.Decoder1, val *interface{}) (err error) {
			return fmt.Errorf("error")
		}),
	}
}
