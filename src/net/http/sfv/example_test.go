// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sfv

import (
	"fmt"
	"log"
)

func ExampleUnmarshalList() {
	v, err := UnmarshalList([]string{`"/member/*/author", "/member/*/comments"`})
	if err != nil {
		log.Fatalln("error: ", err)
	}

	fmt.Println("authors selector: ", v[0].(Item).Value)
	fmt.Println("comments selector: ", v[1].(Item).Value)
	// Output:
	// authors selector:  /member/*/author
	// comments selector:  /member/*/comments
}

func ExampleMarshal() {
	p := List{NewItem("/member/*/author"), NewItem("/member/*/comments")}

	v, err := Marshal(p)
	if err != nil {
		log.Fatalln("error: ", err)
	}

	fmt.Println(v)
	// Output: "/member/*/author", "/member/*/comments"
}
