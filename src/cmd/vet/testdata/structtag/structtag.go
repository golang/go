// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains the test for canonical struct tags.

package structtag

type StructTagTest struct {
	A int "hello" // ERROR "`hello` not compatible with reflect.StructTag.Get: bad syntax for struct tag pair"
}
