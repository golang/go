// errorcheck

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test compiler diagnosis of function missing return statements.
// See issue 65 and golang.org/s/go11return.

package p

type T int

var x interface{}
var c chan int

func external() int // ok

func _() int {
} // ERROR "missing return"

func _() int {
	print(1)
} // ERROR "missing return"

// return is okay
func _() int {
	print(1)
	return 2
}

// goto is okay
func _() int {
L:
	print(1)
	goto L
}

// panic is okay
func _() int {
	print(1)
	panic(2)
}

// but only builtin panic
func _() int {
	var panic = func(int) {}
	print(1)
	panic(2)
} // ERROR "missing return"

// block ending in terminating statement is okay
func _() int {
	{
		print(1)
		return 2
	}
}

// block ending in terminating statement is okay
func _() int {
L:
	{
		print(1)
		goto L
	}
}

// block ending in terminating statement is okay
func _() int {
	print(1)
	{
		panic(2)
	}
}

// adding more code - even though it is dead - now requires a return

func _() int {
	print(1)
	return 2
	print(3)
} // ERROR "missing return"

func _() int {
L:
	print(1)
	goto L
	print(3)
} // ERROR "missing return"

func _() int {
	print(1)
	panic(2)
	print(3)
} // ERROR "missing return"

func _() int {
	{
		print(1)
		return 2
		print(3)
	}
} // ERROR "missing return"

func _() int {
L:
	{
		print(1)
		goto L
		print(3)
	}
} // ERROR "missing return"

func _() int {
	print(1)
	{
		panic(2)
		print(3)
	}
} // ERROR "missing return"

func _() int {
	{
		print(1)
		return 2
	}
	print(3)
} // ERROR "missing return"

func _() int {
L:
	{
		print(1)
		goto L
	}
	print(3)
} // ERROR "missing return"

func _() int {
	print(1)
	{
		panic(2)
	}
	print(3)
} // ERROR "missing return"

// even an empty dead block triggers the message, because it
// becomes the final statement.

func _() int {
	print(1)
	return 2
	{}
} // ERROR "missing return"

func _() int {
L:
	print(1)
	goto L
	{}
} // ERROR "missing return"

func _() int {
	print(1)
	panic(2)
	{}
} // ERROR "missing return"

func _() int {
	{
		print(1)
		return 2
		{}
	}
} // ERROR "missing return"

func _() int {
L:
	{
		print(1)
		goto L
		{}
	}
} // ERROR "missing return"

func _() int {
	print(1)
	{
		panic(2)
		{}
	}
} // ERROR "missing return"

func _() int {
	{
		print(1)
		return 2
	}
	{}
} // ERROR "missing return"

func _() int {
L:
	{
		print(1)
		goto L
	}
	{}
} // ERROR "missing return"

func _() int {
	print(1)
	{
		panic(2)
	}
	{}
} // ERROR "missing return"

// if-else chain with final else and all terminating is okay

func _() int {
	print(1)
	if x == nil {
		panic(2)
	} else {
		panic(3)
	}
}

func _() int {
L:
	print(1)
	if x == nil {
		panic(2)
	} else {
		goto L
	}
}

func _() int {
L:
	print(1)
	if x == nil {
		panic(2)
	} else if x == 1 {
		return 0
	} else if x != 2 {
		panic(3)
	} else {
		goto L
	}
}

// if-else chain missing final else is not okay, even if the
// conditions cover every possible case.

func _() int {
	print(1)
	if x == nil {
		panic(2)
	} else if x != nil {
		panic(3)
	}
} // ERROR "missing return"

func _() int {
	print(1)
	if x == nil {
		panic(2)
	}
} // ERROR "missing return"

func _() int {
	print(1)
	if x == nil {
		panic(2)
	} else if x == 1 {
		return 0
	} else if x != 1 {
		panic(3)
	}
} // ERROR "missing return"


// for { loops that never break are okay.

func _() int {
	print(1)
	for {}
}

func _() int {
	for {
		for {
			break
		}
	}
}

func _() int {
	for {
		L:
		for {
			break L
		}
	}
}

// for { loops that break are not okay.

func _() int {
	print(1)
	for { break }
} // ERROR "missing return"

func _() int {
	for {
		for {
		}
		break
	}
} // ERROR "missing return"

func _() int {
L:
	for {
		for {
			break L
		}
	}
} // ERROR "missing return"

// if there's a condition - even "true" - the loops are no longer syntactically terminating

func _() int {
	print(1)
	for x == nil {}
} // ERROR "missing return"

func _() int {
	for x == nil {
		for {
			break
		}
	}
} // ERROR "missing return"

func _() int {
	for x == nil {
		L:
		for {
			break L
		}
	}	
} // ERROR "missing return"

func _() int {
	print(1)
	for true {}
} // ERROR "missing return"

func _() int {
	for true {
		for {
			break
		}
	}
} // ERROR "missing return"

func _() int {
	for true {
		L:
		for {
			break L
		}
	}
} // ERROR "missing return"

// select in which all cases terminate and none break are okay.

func _() int {
	print(1)
	select{}
}

func _() int {
	print(1)
	select {
	case <-c:
		print(2)
		panic("abc")
	}
}

func _() int {
	print(1)
	select {
	case <-c:
		print(2)
		for{}
	}
}

func _() int {
L:
	print(1)
	select {
	case <-c:
		print(2)
		panic("abc")
	case c <- 1:
		print(2)
		goto L
	}
}

func _() int {
	print(1)
	select {
	case <-c:
		print(2)
		panic("abc")
	default:
		select{}
	}
}

// if any cases don't terminate, the select isn't okay anymore

func _() int {
	print(1)
	select {
	case <-c:
		print(2)
	}
} // ERROR "missing return"

func _() int {
L:
	print(1)
	select {
	case <-c:
		print(2)
		panic("abc")
		goto L
	case c <- 1:
		print(2)
	}
} // ERROR "missing return"


func _() int {
	print(1)
	select {
	case <-c:
		print(2)
		panic("abc")
	default:
		print(2)
	}
} // ERROR "missing return"


// if any breaks refer to the select, the select isn't okay anymore, even if they're dead

func _() int {
	print(1)
	select{ default: break }
} // ERROR "missing return"

func _() int {
	print(1)
	select {
	case <-c:
		print(2)
		panic("abc")
		break
	}
} // ERROR "missing return"

func _() int {
	print(1)
L:
	select {
	case <-c:
		print(2)
		for{ break L }
	}
} // ERROR "missing return"

func _() int {
	print(1)
L:
	select {
	case <-c:
		print(2)
		panic("abc")
	case c <- 1:
		print(2)
		break L
	}
} // ERROR "missing return"

func _() int {
	print(1)
	select {
	case <-c:
		print(1)
		panic("abc")
	default:
		select{}
		break
	}
} // ERROR "missing return"

// switch with default in which all cases terminate is okay

func _() int {
	print(1)
	switch x {
	case 1:
		print(2)
		panic(3)
	default:
		return 4
	}
}

func _() int {
	print(1)
	switch x {
	default:
		return 4
	case 1:
		print(2)
		panic(3)
	}
}

func _() int {
	print(1)
	switch x {
	case 1:
		print(2)
		fallthrough
	default:
		return 4
	}
}

// if no default or some case doesn't terminate, switch is no longer okay

func _() int {
	print(1)
	switch {
	}
} // ERROR "missing return"


func _() int {
	print(1)
	switch x {
	case 1:
		print(2)
		panic(3)
	case 2:
		return 4
	}
} // ERROR "missing return"

func _() int {
	print(1)
	switch x {
	case 2:
		return 4
	case 1:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

func _() int {
	print(1)
	switch x {
	case 1:
		print(2)
		fallthrough
	case 2:
		return 4
	}
} // ERROR "missing return"

func _() int {
	print(1)
	switch x {
	case 1:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

// if any breaks refer to the switch, switch is no longer okay

func _() int {
	print(1)
L:
	switch x {
	case 1:
		print(2)
		panic(3)
		break L
	default:
		return 4
	}
} // ERROR "missing return"

func _() int {
	print(1)
	switch x {
	default:
		return 4
		break
	case 1:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

func _() int {
	print(1)
L:
	switch x {
	case 1:
		print(2)
		for {
			break L
		}
	default:
		return 4
	}
} // ERROR "missing return"

// type switch with default in which all cases terminate is okay

func _() int {
	print(1)
	switch x.(type) {
	case int:
		print(2)
		panic(3)
	default:
		return 4
	}
}

func _() int {
	print(1)
	switch x.(type) {
	default:
		return 4
	case int:
		print(2)
		panic(3)
	}
}

// if no default or some case doesn't terminate, switch is no longer okay

func _() int {
	print(1)
	switch {
	}
} // ERROR "missing return"


func _() int {
	print(1)
	switch x.(type) {
	case int:
		print(2)
		panic(3)
	case float64:
		return 4
	}
} // ERROR "missing return"

func _() int {
	print(1)
	switch x.(type) {
	case float64:
		return 4
	case int:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

func _() int {
	print(1)
	switch x.(type) {
	case int:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

// if any breaks refer to the switch, switch is no longer okay

func _() int {
	print(1)
L:
	switch x.(type) {
	case int:
		print(2)
		panic(3)
		break L
	default:
		return 4
	}
} // ERROR "missing return"

func _() int {
	print(1)
	switch x.(type) {
	default:
		return 4
		break
	case int:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

func _() int {
	print(1)
L:
	switch x.(type) {
	case int:
		print(2)
		for {
			break L
		}
	default:
		return 4
	}
} // ERROR "missing return"

// again, but without the leading print(1).
// testing that everything works when the terminating statement is first.

func _() int {
} // ERROR "missing return"

// return is okay
func _() int {
	return 2
}

// goto is okay
func _() int {
L:
	goto L
}

// panic is okay
func _() int {
	panic(2)
}

// but only builtin panic
func _() int {
	var panic = func(int) {}
	panic(2)
} // ERROR "missing return"

// block ending in terminating statement is okay
func _() int {
	{
		return 2
	}
}

// block ending in terminating statement is okay
func _() int {
L:
	{
		goto L
	}
}

// block ending in terminating statement is okay
func _() int {
	{
		panic(2)
	}
}

// adding more code - even though it is dead - now requires a return

func _() int {
	return 2
	print(3)
} // ERROR "missing return"

func _() int {
L:
	goto L
	print(3)
} // ERROR "missing return"

func _() int {
	panic(2)
	print(3)
} // ERROR "missing return"

func _() int {
	{
		return 2
		print(3)
	}
} // ERROR "missing return"

func _() int {
L:
	{
		goto L
		print(3)
	}
} // ERROR "missing return"

func _() int {
	{
		panic(2)
		print(3)
	}
} // ERROR "missing return"

func _() int {
	{
		return 2
	}
	print(3)
} // ERROR "missing return"

func _() int {
L:
	{
		goto L
	}
	print(3)
} // ERROR "missing return"

func _() int {
	{
		panic(2)
	}
	print(3)
} // ERROR "missing return"

// even an empty dead block triggers the message, because it
// becomes the final statement.

func _() int {
	return 2
	{}
} // ERROR "missing return"

func _() int {
L:
	goto L
	{}
} // ERROR "missing return"

func _() int {
	panic(2)
	{}
} // ERROR "missing return"

func _() int {
	{
		return 2
		{}
	}
} // ERROR "missing return"

func _() int {
L:
	{
		goto L
		{}
	}
} // ERROR "missing return"

func _() int {
	{
		panic(2)
		{}
	}
} // ERROR "missing return"

func _() int {
	{
		return 2
	}
	{}
} // ERROR "missing return"

func _() int {
L:
	{
		goto L
	}
	{}
} // ERROR "missing return"

func _() int {
	{
		panic(2)
	}
	{}
} // ERROR "missing return"

// if-else chain with final else and all terminating is okay

func _() int {
	if x == nil {
		panic(2)
	} else {
		panic(3)
	}
}

func _() int {
L:
	if x == nil {
		panic(2)
	} else {
		goto L
	}
}

func _() int {
L:
	if x == nil {
		panic(2)
	} else if x == 1 {
		return 0
	} else if x != 2 {
		panic(3)
	} else {
		goto L
	}
}

// if-else chain missing final else is not okay, even if the
// conditions cover every possible case.

func _() int {
	if x == nil {
		panic(2)
	} else if x != nil {
		panic(3)
	}
} // ERROR "missing return"

func _() int {
	if x == nil {
		panic(2)
	}
} // ERROR "missing return"

func _() int {
	if x == nil {
		panic(2)
	} else if x == 1 {
		return 0
	} else if x != 1 {
		panic(3)
	}
} // ERROR "missing return"


// for { loops that never break are okay.

func _() int {
	for {}
}

func _() int {
	for {
		for {
			break
		}
	}
}

func _() int {
	for {
		L:
		for {
			break L
		}
	}
}

// for { loops that break are not okay.

func _() int {
	for { break }
} // ERROR "missing return"

func _() int {
	for {
		for {
		}
		break
	}
} // ERROR "missing return"

func _() int {
L:
	for {
		for {
			break L
		}
	}
} // ERROR "missing return"

// if there's a condition - even "true" - the loops are no longer syntactically terminating

func _() int {
	for x == nil {}
} // ERROR "missing return"

func _() int {
	for x == nil {
		for {
			break
		}
	}
} // ERROR "missing return"

func _() int {
	for x == nil {
		L:
		for {
			break L
		}
	}	
} // ERROR "missing return"

func _() int {
	for true {}
} // ERROR "missing return"

func _() int {
	for true {
		for {
			break
		}
	}
} // ERROR "missing return"

func _() int {
	for true {
		L:
		for {
			break L
		}
	}
} // ERROR "missing return"

// select in which all cases terminate and none break are okay.

func _() int {
	select{}
}

func _() int {
	select {
	case <-c:
		print(2)
		panic("abc")
	}
}

func _() int {
	select {
	case <-c:
		print(2)
		for{}
	}
}

func _() int {
L:
	select {
	case <-c:
		print(2)
		panic("abc")
	case c <- 1:
		print(2)
		goto L
	}
}

func _() int {
	select {
	case <-c:
		print(2)
		panic("abc")
	default:
		select{}
	}
}

// if any cases don't terminate, the select isn't okay anymore

func _() int {
	select {
	case <-c:
		print(2)
	}
} // ERROR "missing return"

func _() int {
L:
	select {
	case <-c:
		print(2)
		panic("abc")
		goto L
	case c <- 1:
		print(2)
	}
} // ERROR "missing return"


func _() int {
	select {
	case <-c:
		print(2)
		panic("abc")
	default:
		print(2)
	}
} // ERROR "missing return"


// if any breaks refer to the select, the select isn't okay anymore, even if they're dead

func _() int {
	select{ default: break }
} // ERROR "missing return"

func _() int {
	select {
	case <-c:
		print(2)
		panic("abc")
		break
	}
} // ERROR "missing return"

func _() int {
L:
	select {
	case <-c:
		print(2)
		for{ break L }
	}
} // ERROR "missing return"

func _() int {
L:
	select {
	case <-c:
		print(2)
		panic("abc")
	case c <- 1:
		print(2)
		break L
	}
} // ERROR "missing return"

func _() int {
	select {
	case <-c:
		panic("abc")
	default:
		select{}
		break
	}
} // ERROR "missing return"

// switch with default in which all cases terminate is okay

func _() int {
	switch x {
	case 1:
		print(2)
		panic(3)
	default:
		return 4
	}
}

func _() int {
	switch x {
	default:
		return 4
	case 1:
		print(2)
		panic(3)
	}
}

func _() int {
	switch x {
	case 1:
		print(2)
		fallthrough
	default:
		return 4
	}
}

// if no default or some case doesn't terminate, switch is no longer okay

func _() int {
	switch {
	}
} // ERROR "missing return"


func _() int {
	switch x {
	case 1:
		print(2)
		panic(3)
	case 2:
		return 4
	}
} // ERROR "missing return"

func _() int {
	switch x {
	case 2:
		return 4
	case 1:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

func _() int {
	switch x {
	case 1:
		print(2)
		fallthrough
	case 2:
		return 4
	}
} // ERROR "missing return"

func _() int {
	switch x {
	case 1:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

// if any breaks refer to the switch, switch is no longer okay

func _() int {
L:
	switch x {
	case 1:
		print(2)
		panic(3)
		break L
	default:
		return 4
	}
} // ERROR "missing return"

func _() int {
	switch x {
	default:
		return 4
		break
	case 1:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

func _() int {
L:
	switch x {
	case 1:
		print(2)
		for {
			break L
		}
	default:
		return 4
	}
} // ERROR "missing return"

// type switch with default in which all cases terminate is okay

func _() int {
	switch x.(type) {
	case int:
		print(2)
		panic(3)
	default:
		return 4
	}
}

func _() int {
	switch x.(type) {
	default:
		return 4
	case int:
		print(2)
		panic(3)
	}
}

// if no default or some case doesn't terminate, switch is no longer okay

func _() int {
	switch {
	}
} // ERROR "missing return"


func _() int {
	switch x.(type) {
	case int:
		print(2)
		panic(3)
	case float64:
		return 4
	}
} // ERROR "missing return"

func _() int {
	switch x.(type) {
	case float64:
		return 4
	case int:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

func _() int {
	switch x.(type) {
	case int:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

// if any breaks refer to the switch, switch is no longer okay

func _() int {
L:
	switch x.(type) {
	case int:
		print(2)
		panic(3)
		break L
	default:
		return 4
	}
} // ERROR "missing return"

func _() int {
	switch x.(type) {
	default:
		return 4
		break
	case int:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

func _() int {
L:
	switch x.(type) {
	case int:
		print(2)
		for {
			break L
		}
	default:
		return 4
	}
} // ERROR "missing return"

func _() int {
	switch x.(type) {
	default:
		return 4
	case int, float64:
		print(2)
		panic(3)
	}
}

// again, with func literals

var _ = func() int {
} // ERROR "missing return"

var _ = func() int {
	print(1)
} // ERROR "missing return"

// return is okay
var _ = func() int {
	print(1)
	return 2
}

// goto is okay
var _ = func() int {
L:
	print(1)
	goto L
}

// panic is okay
var _ = func() int {
	print(1)
	panic(2)
}

// but only builtin panic
var _ = func() int {
	var panic = func(int) {}
	print(1)
	panic(2)
} // ERROR "missing return"

// block ending in terminating statement is okay
var _ = func() int {
	{
		print(1)
		return 2
	}
}

// block ending in terminating statement is okay
var _ = func() int {
L:
	{
		print(1)
		goto L
	}
}

// block ending in terminating statement is okay
var _ = func() int {
	print(1)
	{
		panic(2)
	}
}

// adding more code - even though it is dead - now requires a return

var _ = func() int {
	print(1)
	return 2
	print(3)
} // ERROR "missing return"

var _ = func() int {
L:
	print(1)
	goto L
	print(3)
} // ERROR "missing return"

var _ = func() int {
	print(1)
	panic(2)
	print(3)
} // ERROR "missing return"

var _ = func() int {
	{
		print(1)
		return 2
		print(3)
	}
} // ERROR "missing return"

var _ = func() int {
L:
	{
		print(1)
		goto L
		print(3)
	}
} // ERROR "missing return"

var _ = func() int {
	print(1)
	{
		panic(2)
		print(3)
	}
} // ERROR "missing return"

var _ = func() int {
	{
		print(1)
		return 2
	}
	print(3)
} // ERROR "missing return"

var _ = func() int {
L:
	{
		print(1)
		goto L
	}
	print(3)
} // ERROR "missing return"

var _ = func() int {
	print(1)
	{
		panic(2)
	}
	print(3)
} // ERROR "missing return"

// even an empty dead block triggers the message, because it
// becomes the final statement.

var _ = func() int {
	print(1)
	return 2
	{}
} // ERROR "missing return"

var _ = func() int {
L:
	print(1)
	goto L
	{}
} // ERROR "missing return"

var _ = func() int {
	print(1)
	panic(2)
	{}
} // ERROR "missing return"

var _ = func() int {
	{
		print(1)
		return 2
		{}
	}
} // ERROR "missing return"

var _ = func() int {
L:
	{
		print(1)
		goto L
		{}
	}
} // ERROR "missing return"

var _ = func() int {
	print(1)
	{
		panic(2)
		{}
	}
} // ERROR "missing return"

var _ = func() int {
	{
		print(1)
		return 2
	}
	{}
} // ERROR "missing return"

var _ = func() int {
L:
	{
		print(1)
		goto L
	}
	{}
} // ERROR "missing return"

var _ = func() int {
	print(1)
	{
		panic(2)
	}
	{}
} // ERROR "missing return"

// if-else chain with final else and all terminating is okay

var _ = func() int {
	print(1)
	if x == nil {
		panic(2)
	} else {
		panic(3)
	}
}

var _ = func() int {
L:
	print(1)
	if x == nil {
		panic(2)
	} else {
		goto L
	}
}

var _ = func() int {
L:
	print(1)
	if x == nil {
		panic(2)
	} else if x == 1 {
		return 0
	} else if x != 2 {
		panic(3)
	} else {
		goto L
	}
}

// if-else chain missing final else is not okay, even if the
// conditions cover every possible case.

var _ = func() int {
	print(1)
	if x == nil {
		panic(2)
	} else if x != nil {
		panic(3)
	}
} // ERROR "missing return"

var _ = func() int {
	print(1)
	if x == nil {
		panic(2)
	}
} // ERROR "missing return"

var _ = func() int {
	print(1)
	if x == nil {
		panic(2)
	} else if x == 1 {
		return 0
	} else if x != 1 {
		panic(3)
	}
} // ERROR "missing return"


// for { loops that never break are okay.

var _ = func() int {
	print(1)
	for {}
}

var _ = func() int {
	for {
		for {
			break
		}
	}
}

var _ = func() int {
	for {
		L:
		for {
			break L
		}
	}
}

// for { loops that break are not okay.

var _ = func() int {
	print(1)
	for { break }
} // ERROR "missing return"

var _ = func() int {
	for {
		for {
		}
		break
	}
} // ERROR "missing return"

var _ = func() int {
L:
	for {
		for {
			break L
		}
	}
} // ERROR "missing return"

// if there's a condition - even "true" - the loops are no longer syntactically terminating

var _ = func() int {
	print(1)
	for x == nil {}
} // ERROR "missing return"

var _ = func() int {
	for x == nil {
		for {
			break
		}
	}
} // ERROR "missing return"

var _ = func() int {
	for x == nil {
		L:
		for {
			break L
		}
	}	
} // ERROR "missing return"

var _ = func() int {
	print(1)
	for true {}
} // ERROR "missing return"

var _ = func() int {
	for true {
		for {
			break
		}
	}
} // ERROR "missing return"

var _ = func() int {
	for true {
		L:
		for {
			break L
		}
	}
} // ERROR "missing return"

// select in which all cases terminate and none break are okay.

var _ = func() int {
	print(1)
	select{}
}

var _ = func() int {
	print(1)
	select {
	case <-c:
		print(2)
		panic("abc")
	}
}

var _ = func() int {
	print(1)
	select {
	case <-c:
		print(2)
		for{}
	}
}

var _ = func() int {
L:
	print(1)
	select {
	case <-c:
		print(2)
		panic("abc")
	case c <- 1:
		print(2)
		goto L
	}
}

var _ = func() int {
	print(1)
	select {
	case <-c:
		print(2)
		panic("abc")
	default:
		select{}
	}
}

// if any cases don't terminate, the select isn't okay anymore

var _ = func() int {
	print(1)
	select {
	case <-c:
		print(2)
	}
} // ERROR "missing return"

var _ = func() int {
L:
	print(1)
	select {
	case <-c:
		print(2)
		panic("abc")
		goto L
	case c <- 1:
		print(2)
	}
} // ERROR "missing return"


var _ = func() int {
	print(1)
	select {
	case <-c:
		print(2)
		panic("abc")
	default:
		print(2)
	}
} // ERROR "missing return"


// if any breaks refer to the select, the select isn't okay anymore, even if they're dead

var _ = func() int {
	print(1)
	select{ default: break }
} // ERROR "missing return"

var _ = func() int {
	print(1)
	select {
	case <-c:
		print(2)
		panic("abc")
		break
	}
} // ERROR "missing return"

var _ = func() int {
	print(1)
L:
	select {
	case <-c:
		print(2)
		for{ break L }
	}
} // ERROR "missing return"

var _ = func() int {
	print(1)
L:
	select {
	case <-c:
		print(2)
		panic("abc")
	case c <- 1:
		print(2)
		break L
	}
} // ERROR "missing return"

var _ = func() int {
	print(1)
	select {
	case <-c:
		print(1)
		panic("abc")
	default:
		select{}
		break
	}
} // ERROR "missing return"

// switch with default in which all cases terminate is okay

var _ = func() int {
	print(1)
	switch x {
	case 1:
		print(2)
		panic(3)
	default:
		return 4
	}
}

var _ = func() int {
	print(1)
	switch x {
	default:
		return 4
	case 1:
		print(2)
		panic(3)
	}
}

var _ = func() int {
	print(1)
	switch x {
	case 1:
		print(2)
		fallthrough
	default:
		return 4
	}
}

// if no default or some case doesn't terminate, switch is no longer okay

var _ = func() int {
	print(1)
	switch {
	}
} // ERROR "missing return"


var _ = func() int {
	print(1)
	switch x {
	case 1:
		print(2)
		panic(3)
	case 2:
		return 4
	}
} // ERROR "missing return"

var _ = func() int {
	print(1)
	switch x {
	case 2:
		return 4
	case 1:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

var _ = func() int {
	print(1)
	switch x {
	case 1:
		print(2)
		fallthrough
	case 2:
		return 4
	}
} // ERROR "missing return"

var _ = func() int {
	print(1)
	switch x {
	case 1:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

// if any breaks refer to the switch, switch is no longer okay

var _ = func() int {
	print(1)
L:
	switch x {
	case 1:
		print(2)
		panic(3)
		break L
	default:
		return 4
	}
} // ERROR "missing return"

var _ = func() int {
	print(1)
	switch x {
	default:
		return 4
		break
	case 1:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

var _ = func() int {
	print(1)
L:
	switch x {
	case 1:
		print(2)
		for {
			break L
		}
	default:
		return 4
	}
} // ERROR "missing return"

// type switch with default in which all cases terminate is okay

var _ = func() int {
	print(1)
	switch x.(type) {
	case int:
		print(2)
		panic(3)
	default:
		return 4
	}
}

var _ = func() int {
	print(1)
	switch x.(type) {
	default:
		return 4
	case int:
		print(2)
		panic(3)
	}
}

// if no default or some case doesn't terminate, switch is no longer okay

var _ = func() int {
	print(1)
	switch {
	}
} // ERROR "missing return"


var _ = func() int {
	print(1)
	switch x.(type) {
	case int:
		print(2)
		panic(3)
	case float64:
		return 4
	}
} // ERROR "missing return"

var _ = func() int {
	print(1)
	switch x.(type) {
	case float64:
		return 4
	case int:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

var _ = func() int {
	print(1)
	switch x.(type) {
	case int:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

// if any breaks refer to the switch, switch is no longer okay

var _ = func() int {
	print(1)
L:
	switch x.(type) {
	case int:
		print(2)
		panic(3)
		break L
	default:
		return 4
	}
} // ERROR "missing return"

var _ = func() int {
	print(1)
	switch x.(type) {
	default:
		return 4
		break
	case int:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

var _ = func() int {
	print(1)
L:
	switch x.(type) {
	case int:
		print(2)
		for {
			break L
		}
	default:
		return 4
	}
} // ERROR "missing return"

// again, but without the leading print(1).
// testing that everything works when the terminating statement is first.

var _ = func() int {
} // ERROR "missing return"

// return is okay
var _ = func() int {
	return 2
}

// goto is okay
var _ = func() int {
L:
	goto L
}

// panic is okay
var _ = func() int {
	panic(2)
}

// but only builtin panic
var _ = func() int {
	var panic = func(int) {}
	panic(2)
} // ERROR "missing return"

// block ending in terminating statement is okay
var _ = func() int {
	{
		return 2
	}
}

// block ending in terminating statement is okay
var _ = func() int {
L:
	{
		goto L
	}
}

// block ending in terminating statement is okay
var _ = func() int {
	{
		panic(2)
	}
}

// adding more code - even though it is dead - now requires a return

var _ = func() int {
	return 2
	print(3)
} // ERROR "missing return"

var _ = func() int {
L:
	goto L
	print(3)
} // ERROR "missing return"

var _ = func() int {
	panic(2)
	print(3)
} // ERROR "missing return"

var _ = func() int {
	{
		return 2
		print(3)
	}
} // ERROR "missing return"

var _ = func() int {
L:
	{
		goto L
		print(3)
	}
} // ERROR "missing return"

var _ = func() int {
	{
		panic(2)
		print(3)
	}
} // ERROR "missing return"

var _ = func() int {
	{
		return 2
	}
	print(3)
} // ERROR "missing return"

var _ = func() int {
L:
	{
		goto L
	}
	print(3)
} // ERROR "missing return"

var _ = func() int {
	{
		panic(2)
	}
	print(3)
} // ERROR "missing return"

// even an empty dead block triggers the message, because it
// becomes the final statement.

var _ = func() int {
	return 2
	{}
} // ERROR "missing return"

var _ = func() int {
L:
	goto L
	{}
} // ERROR "missing return"

var _ = func() int {
	panic(2)
	{}
} // ERROR "missing return"

var _ = func() int {
	{
		return 2
		{}
	}
} // ERROR "missing return"

var _ = func() int {
L:
	{
		goto L
		{}
	}
} // ERROR "missing return"

var _ = func() int {
	{
		panic(2)
		{}
	}
} // ERROR "missing return"

var _ = func() int {
	{
		return 2
	}
	{}
} // ERROR "missing return"

var _ = func() int {
L:
	{
		goto L
	}
	{}
} // ERROR "missing return"

var _ = func() int {
	{
		panic(2)
	}
	{}
} // ERROR "missing return"

// if-else chain with final else and all terminating is okay

var _ = func() int {
	if x == nil {
		panic(2)
	} else {
		panic(3)
	}
}

var _ = func() int {
L:
	if x == nil {
		panic(2)
	} else {
		goto L
	}
}

var _ = func() int {
L:
	if x == nil {
		panic(2)
	} else if x == 1 {
		return 0
	} else if x != 2 {
		panic(3)
	} else {
		goto L
	}
}

// if-else chain missing final else is not okay, even if the
// conditions cover every possible case.

var _ = func() int {
	if x == nil {
		panic(2)
	} else if x != nil {
		panic(3)
	}
} // ERROR "missing return"

var _ = func() int {
	if x == nil {
		panic(2)
	}
} // ERROR "missing return"

var _ = func() int {
	if x == nil {
		panic(2)
	} else if x == 1 {
		return 0
	} else if x != 1 {
		panic(3)
	}
} // ERROR "missing return"


// for { loops that never break are okay.

var _ = func() int {
	for {}
}

var _ = func() int {
	for {
		for {
			break
		}
	}
}

var _ = func() int {
	for {
		L:
		for {
			break L
		}
	}
}

// for { loops that break are not okay.

var _ = func() int {
	for { break }
} // ERROR "missing return"

var _ = func() int {
	for {
		for {
		}
		break
	}
} // ERROR "missing return"

var _ = func() int {
L:
	for {
		for {
			break L
		}
	}
} // ERROR "missing return"

// if there's a condition - even "true" - the loops are no longer syntactically terminating

var _ = func() int {
	for x == nil {}
} // ERROR "missing return"

var _ = func() int {
	for x == nil {
		for {
			break
		}
	}
} // ERROR "missing return"

var _ = func() int {
	for x == nil {
		L:
		for {
			break L
		}
	}	
} // ERROR "missing return"

var _ = func() int {
	for true {}
} // ERROR "missing return"

var _ = func() int {
	for true {
		for {
			break
		}
	}
} // ERROR "missing return"

var _ = func() int {
	for true {
		L:
		for {
			break L
		}
	}
} // ERROR "missing return"

// select in which all cases terminate and none break are okay.

var _ = func() int {
	select{}
}

var _ = func() int {
	select {
	case <-c:
		print(2)
		panic("abc")
	}
}

var _ = func() int {
	select {
	case <-c:
		print(2)
		for{}
	}
}

var _ = func() int {
L:
	select {
	case <-c:
		print(2)
		panic("abc")
	case c <- 1:
		print(2)
		goto L
	}
}

var _ = func() int {
	select {
	case <-c:
		print(2)
		panic("abc")
	default:
		select{}
	}
}

// if any cases don't terminate, the select isn't okay anymore

var _ = func() int {
	select {
	case <-c:
		print(2)
	}
} // ERROR "missing return"

var _ = func() int {
L:
	select {
	case <-c:
		print(2)
		panic("abc")
		goto L
	case c <- 1:
		print(2)
	}
} // ERROR "missing return"


var _ = func() int {
	select {
	case <-c:
		print(2)
		panic("abc")
	default:
		print(2)
	}
} // ERROR "missing return"


// if any breaks refer to the select, the select isn't okay anymore, even if they're dead

var _ = func() int {
	select{ default: break }
} // ERROR "missing return"

var _ = func() int {
	select {
	case <-c:
		print(2)
		panic("abc")
		break
	}
} // ERROR "missing return"

var _ = func() int {
L:
	select {
	case <-c:
		print(2)
		for{ break L }
	}
} // ERROR "missing return"

var _ = func() int {
L:
	select {
	case <-c:
		print(2)
		panic("abc")
	case c <- 1:
		print(2)
		break L
	}
} // ERROR "missing return"

var _ = func() int {
	select {
	case <-c:
		panic("abc")
	default:
		select{}
		break
	}
} // ERROR "missing return"

// switch with default in which all cases terminate is okay

var _ = func() int {
	switch x {
	case 1:
		print(2)
		panic(3)
	default:
		return 4
	}
}

var _ = func() int {
	switch x {
	default:
		return 4
	case 1:
		print(2)
		panic(3)
	}
}

var _ = func() int {
	switch x {
	case 1:
		print(2)
		fallthrough
	default:
		return 4
	}
}

// if no default or some case doesn't terminate, switch is no longer okay

var _ = func() int {
	switch {
	}
} // ERROR "missing return"


var _ = func() int {
	switch x {
	case 1:
		print(2)
		panic(3)
	case 2:
		return 4
	}
} // ERROR "missing return"

var _ = func() int {
	switch x {
	case 2:
		return 4
	case 1:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

var _ = func() int {
	switch x {
	case 1:
		print(2)
		fallthrough
	case 2:
		return 4
	}
} // ERROR "missing return"

var _ = func() int {
	switch x {
	case 1:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

// if any breaks refer to the switch, switch is no longer okay

var _ = func() int {
L:
	switch x {
	case 1:
		print(2)
		panic(3)
		break L
	default:
		return 4
	}
} // ERROR "missing return"

var _ = func() int {
	switch x {
	default:
		return 4
		break
	case 1:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

var _ = func() int {
L:
	switch x {
	case 1:
		print(2)
		for {
			break L
		}
	default:
		return 4
	}
} // ERROR "missing return"

// type switch with default in which all cases terminate is okay

var _ = func() int {
	switch x.(type) {
	case int:
		print(2)
		panic(3)
	default:
		return 4
	}
}

var _ = func() int {
	switch x.(type) {
	default:
		return 4
	case int:
		print(2)
		panic(3)
	}
}

// if no default or some case doesn't terminate, switch is no longer okay

var _ = func() int {
	switch {
	}
} // ERROR "missing return"


var _ = func() int {
	switch x.(type) {
	case int:
		print(2)
		panic(3)
	case float64:
		return 4
	}
} // ERROR "missing return"

var _ = func() int {
	switch x.(type) {
	case float64:
		return 4
	case int:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

var _ = func() int {
	switch x.(type) {
	case int:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

// if any breaks refer to the switch, switch is no longer okay

var _ = func() int {
L:
	switch x.(type) {
	case int:
		print(2)
		panic(3)
		break L
	default:
		return 4
	}
} // ERROR "missing return"

var _ = func() int {
	switch x.(type) {
	default:
		return 4
		break
	case int:
		print(2)
		panic(3)
	}
} // ERROR "missing return"

var _ = func() int {
L:
	switch x.(type) {
	case int:
		print(2)
		for {
			break L
		}
	default:
		return 4
	}
} // ERROR "missing return"

var _ = func() int {
	switch x.(type) {
	default:
		return 4
	case int, float64:
		print(2)
		panic(3)
	}
}

/**/
