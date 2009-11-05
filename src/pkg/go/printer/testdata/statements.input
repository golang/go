// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package statements

var expr bool;

func use(x interface{}) {}

// Formatting of if-statement headers.
func _() {
	if {}
	if;{}  // no semicolon printed
	if expr{}
	if;expr{}  // no semicolon printed
	if (expr){}  // no parens printed
	if;((expr)){}  // no semicolon and parens printed
	if x:=expr;{
	use(x)}
	if x:=expr; expr {use(x)}
}


// Formatting of switch-statement headers.
func _() {
	switch {}
	switch;{}  // no semicolon printed
	switch expr {}
	switch;expr{}  // no semicolon printed
	switch (expr) {}  // no parens printed
	switch;((expr)){}  // no semicolon and parens printed
	switch x := expr; { default:use(
x)
	}
	switch x := expr; expr {default:use(x)}
}


// Formatting of switch statement bodies.
func _() {
	switch {
	}

	switch x := 0; x {
	case 1:
		use(x);
		use(x);  // followed by an empty line

	case 2:  // followed by an empty line

		use(x);  // followed by an empty line

	case 3:  // no empty lines
		use(x);
		use(x);
	}

	switch x {
	case 0:
		use(x);
	case 1:  // this comment should have no effect on the previous or next line
		use(x);
	}

	switch x := 0; x {
	case 1:
		x = 0;
		// this comment should be indented
	case 2:
		x = 0;
	// this comment should not be indented, it is aligned with the next case
	case 3:
		x = 0;
		/* indented comment
		   aligned
		   aligned
		*/
		// bla
		/* and more */
	case 4:
		x = 0;
	/* not indented comment
	   aligned
	   aligned
	*/
	// bla
	/* and more */
	case 5:
	}
}


// Formatting of for-statement headers.
func _() {
	for{}
	for expr {}
	for (expr) {}  // no parens printed
	for;;{}  // no semicolons printed
	for x :=expr;; {use( x)}
	for; expr;{}  // no semicolons printed
	for; ((expr));{}  // no semicolons and parens printed
	for; ; expr = false {}
	for x :=expr; expr; {use(x)}
	for x := expr;; expr=false {use(x)}
	for;expr;expr =false {
	}
	for x := expr;expr;expr = false { use(x) }
	for x := range []int{} { use(x) }
}


// Extra empty lines inside functions. Do respect source code line
// breaks between statement boundaries but print at most one empty
// line at a time.
func _() {

	const _ = 0;

	const _ = 1;
	type _ int;
	type _ float;

	var _ = 0;
	var x = 1;

	// Each use(x) call below should have at most one empty line before and after.



	use(x);

	if x < x {

		use(x);

	} else {

		use(x);

	}
}


// Formatting around labels.
func _() {
	L:
}


func _() {
	// this comment should be indented
	L:
}


func _() {
	L: _ = 0;
}


func _() {
	// this comment should be indented
	L: _ = 0;
}


func _() {
	for {
	L1: _ = 0;
	L2:
		_ = 0;
	}
}


func _() {
		// this comment should be indented
	for {
	L1: _ = 0;
	L2:
		_ = 0;
	}
}


func _() {
	if {
		_ = 0;
	}
	_ = 0;  // the indentation here should not be affected by the long label name
AnOverlongLabel:
	_ = 0;
	
	if {
		_ = 0;
	}
	_ = 0;

L:	_ = 0;
}
