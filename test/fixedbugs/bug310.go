// errchk $G $D/$F.go

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import (
	"bytes"
	"fmt"
)

type t int

func main() {
	_ = t.bar	// ERROR "no method"
	var b bytes.Buffer
	fmt.Print(b)	// ERROR "implicit assignment"
}
