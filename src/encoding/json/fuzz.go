// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gofuzz

package json

import (
	"fmt"
)

// Fuzz the input data is a byte slice, The function must return 1 if the
// fuzzer should increase priority of the given input during subsequent
// fuzzing (for example, the input is lexically correct and was parsed successfully);
// -1 if the input must not be added to corpus even if gives new coverage;
// and 0 otherwise;
func Fuzz(data []byte) (score int) {
	for _, ctor := range []func() any{
		func() any { return new(any) },
		func() any { return new(map[string]any) },
		func() any { return new([]any) },
	} {
		v := ctor()
		err := Unmarshal(data, v)
		if err != nil {
			continue
		}
		score = 1

		m, err := Marshal(v)
		if err != nil {
			fmt.Printf("v=%#v\n", v)
			panic(err)
		}

		u := ctor()
		err = Unmarshal(m, u)
		if err != nil {
			fmt.Printf("v=%#v\n", v)
			fmt.Printf("m=%s\n", m)
			panic(err)
		}
	}

	return
}
