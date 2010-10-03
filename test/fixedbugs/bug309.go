// $G $D/$F.go

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 1016

package main

func foo(t interface{}, c chan int) {
	switch v := t.(type) {
	case int:
		select {
		case <-c:
			// bug was: internal compiler error: var without type, init: v
		}
	}
}
