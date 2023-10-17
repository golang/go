// compile

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package foo
import "fmt"

func f() {
	fmt.Println();
	fmt := 1;
	_ = fmt;
}
