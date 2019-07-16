// runoutput

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Generate test of strength reduction for multiplications
// with contstants. Especially useful for amd64/386.

package main

import "fmt"

func testMul(fact, bits int) string {
	n := fmt.Sprintf("testMul_%d_%d", fact, bits)
	fmt.Printf("func %s(s int%d) {\n", n, bits)

	want := 0
	for i := 0; i < 200; i++ {
		fmt.Printf(`	if want, got := int%d(%d), s*%d; want != got {
		failed = true
		fmt.Printf("got %d * %%d == %%d, wanted %d\n",  s, got)
	}
`, bits, want, i, i, want)
		want += fact
	}

	fmt.Printf("}\n")
	return fmt.Sprintf("%s(%d)", n, fact)
}

func main() {
	fmt.Printf("package main\n")
	fmt.Printf("import \"fmt\"\n")
	fmt.Printf("var failed = false\n")

	f1 := testMul(17, 32)
	f2 := testMul(131, 64)

	fmt.Printf("func main() {\n")
	fmt.Println(f1)
	fmt.Println(f2)
	fmt.Printf("if failed {\n	panic(\"multiplication failed\")\n}\n")
	fmt.Printf("}\n")
}
