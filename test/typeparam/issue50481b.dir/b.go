// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package b

import "fmt"

type Foo[T1 ~string, T2 ~int] struct {
	ValueA T1
	ValueB T2
}

func (f *Foo[_, _]) String() string {
	return fmt.Sprintf("%v %v", f.ValueA, f.ValueB)
}
