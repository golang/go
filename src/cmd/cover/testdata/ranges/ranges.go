package main

import "fmt"

func main() {
«	fmt.Println("Start")
	x := 42
»
	// This is a regular comment
«	y := 0
	fmt.Println("After comment")
»
	// Multiple comment lines
	// fmt.Println("commented code")
	// TODO: implement this later

«	fmt.Println("After multiple comments")
»
	/* block comment */

«	fmt.Println("After block comment")
	z := 0
»
	«if x > 0 {»
«		y = x * 2
»	} else {
«		y = x - 2
»	}

«	z = 5
»
	/* Multiline block
	comment spanning
	several lines */

«	z1 := 0
»
«	z1 = 1 /* inline comment
»	spanning lines
«	end */ z1 = 2
»
«	z1 = 3; /* // */ z1 = 4
»
«	z1 = 5 /* //
»	//
«	// */ z1 = 6
»
	/*
«	*/ z1 = 7 /*
»	*/

«	z1 = 8/*
»	*/ /* comment
«	*/z1 = 9
»
«	/* before */ z1 = 10
	/* before */ z1 = 10 /* after */
	             z1 = 10 /* after */
»
«	fmt.Printf("Result: %d\n", z)
	fmt.Printf("Result: %d\n", z1)
»
«	s := `This is a multi-line raw string
	// fake comment on line 2
	/* and fake comment on line 3 */
	and other`
»
«	s = `another multiline string
	` // another trap
»
«	fmt.Printf("%s", s)
»
	// More comments to exclude
	// for i := 0; i < 10; i++ {
	//   fmt.Printf("Loop %d", i)
	// }

«	fmt.Printf("Result: %d\n", y)»
	// end comment
}

func empty() {

}

func singleBlock() {
«	fmt.Printf("ResultSomething")
»}

func justComment() {
	// comment
}

func justMultilineComment() {
	/* comment
	again
	until here */
}

func constBlock() {
«	const (
		A = 1

		B = 2
	)
	fmt.Printf("A=%d B=%d", A, B)
»}

func compositeLit() {
«	m := map[string]int{
		"a": 1,
»	}
«	fmt.Println(m)
»}
