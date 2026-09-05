// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax_test

import (
	"fmt"
	"regexp/syntax"
)

func ExampleParse() {
	re, err := syntax.Parse(`(go|lang)+`, syntax.Perl)
	if err != nil {
		panic(err)
	}

	fmt.Println("String:", re.String())
	fmt.Println("Op:", re.Op)
	fmt.Println("Sub expressions:", len(re.Sub))

	// Walk the public parse tree to inspect its structure.
	var printTree func(*syntax.Regexp, string)
	printTree = func(re *syntax.Regexp, indent string) {
		fmt.Printf("%s%s: %s\n", indent, re.Op, re)
		for _, sub := range re.Sub {
			printTree(sub, indent+"  ")
		}
	}
	printTree(re, "")

	// Output:
	// String: (go|lang)+
	// Op: Plus
	// Sub expressions: 1
	// Plus: (go|lang)+
	//   Capture: (go|lang)
	//     Alternate: go|lang
	//       Literal: go
	//       Literal: lang
}
