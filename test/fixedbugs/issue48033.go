// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strings"
)

type app struct {
	Name string
}

func bug() func() {
	return func() {

		// the issue is this if true block
		if true {
			return
		}

		var xx = []app{}
		var gapp app
		for _, app := range xx {
			if strings.ToUpper("") == app.Name {
				fmt.Printf("%v\n", app)
				gapp = app
			}
		}
		fmt.Println(gapp)
	}
}

func main() {
	bug()
}
