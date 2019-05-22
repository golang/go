// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

type container struct {
	Value string
}

func main() {
	s := []container{
		7: {Value: "string value"},
	}
	if s[7].Value != "string value" {
		panic(fmt.Errorf("wanted \"string value\", got \"%s\"", s[7].Value))
	}
}
