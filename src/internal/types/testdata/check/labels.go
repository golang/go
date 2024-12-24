// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file is a modified concatenation of the files
// $GOROOT/test/label.go and $GOROOT/test/label1.go.

package labels

var x int

func f0() {
L1 /* ERROR "label L1 declared and not used" */ :
	for {
	}
L2 /* ERROR "label L2 declared and not used" */ :
	select {
	}
L3 /* ERROR "label L3 declared and not used" */ :
	switch {
	}
L4 /* ERROR "label L4 declared and not used" */ :
	if true {
	}
L5 /* ERROR "label L5 declared and not used" */ :
	f0()
L6:
	f0()
L6 /* ERROR "label L6 already declared" */ :
	f0()
	if x == 20 {
		goto L6
	}

L7:
	for {
		break L7
		break L8 /* ERROR "invalid break label L8" */
	}

// A label must be directly associated with a switch, select, or
// for statement; it cannot be the label of a labeled statement.

L7a /* ERROR "declared and not used" */ : L7b:
	for {
		break L7a /* ERROR "invalid break label L7a" */
		continue L7a /* ERROR "invalid continue label L7a" */
		continue L7b
	}

L8:
	for {
		if x == 21 {
			continue L8
			continue L7 /* ERROR "invalid continue label L7" */
		}
	}

L9:
	switch {
	case true:
		break L9
	defalt /* ERROR "label defalt declared and not used" */ :
	}

L10:
	select {
	default:
		break L10
		break L9 /* ERROR "invalid break label L9" */
	}

	goto L10a
L10a: L10b:
	select {
	default:
		break L10a /* ERROR "invalid break label L10a" */
		break L10b
		continue L10b /* ERROR "invalid continue label L10b" */
	}
}

func f1() {
L1:
	for {
		if x == 0 {
			break L1
		}
		if x == 1 {
			continue L1
		}
		goto L1
	}

L2:
	select {
	default:
		if x == 0 {
			break L2
		}
		if x == 1 {
			continue L2 /* ERROR "invalid continue label L2" */
		}
		goto L2
	}

L3:
	switch {
	case x > 10:
		if x == 11 {
			break L3
		}
		if x == 12 {
			continue L3 /* ERROR "invalid continue label L3" */
		}
		goto L3
	}

L4:
	if true {
		if x == 13 {
			break L4 /* ERROR "invalid break label L4" */
		}
		if x == 14 {
			continue L4 /* ERROR "invalid continue label L4" */
		}
		if x == 15 {
			goto L4
		}
	}

L5:
	f1()
	if x == 16 {
		break L5 /* ERROR "invalid break label L5" */
	}
	if x == 17 {
		continue L5 /* ERROR "invalid continue label L5" */
	}
	if x == 18 {
		goto L5
	}

	for {
		if x == 19 {
			break L1 /* ERROR "invalid break label L1" */
		}
		if x == 20 {
			continue L1 /* ERROR "invalid continue label L1" */
		}
		if x == 21 {
			goto L1
		}
	}
}

// Additional tests not in the original files.

func f2() {
L1 /* ERROR "label L1 declared and not used" */ :
	if x == 0 {
		for {
			continue L1 /* ERROR "invalid continue label L1" */
		}
	}
}

func f3() {
L1:
L2:
L3:
	for {
		break L1 /* ERROR "invalid break label L1" */
		break L2 /* ERROR "invalid break label L2" */
		break L3
		continue L1 /* ERROR "invalid continue label L1" */
		continue L2 /* ERROR "invalid continue label L2" */
		continue L3
		goto L1
		goto L2
		goto L3
	}
}

// Blank labels are never declared.

func f4() {
_:
_: // multiple blank labels are ok
	goto _ /* ERROR "label _ not declared" */
}

func f5() {
_:
	for {
		break _ /* ERROR "invalid break label _" */
		continue _ /* ERROR "invalid continue label _" */
	}
}

func f6() {
_:
	switch {
	default:
		break _ /* ERROR "invalid break label _" */
	}
}
