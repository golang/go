// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Simple test: enumeration of type int starting at 0.

package main

import "fmt"

type Day int

const (
	Monday Day = iota
	Tuesday
	Wednesday
	Thursday
	Friday
	Saturday
	Sunday
)

func main() {
	ck(Monday, "Monday")
	ck(Tuesday, "Tuesday")
	ck(Wednesday, "Wednesday")
	ck(Thursday, "Thursday")
	ck(Friday, "Friday")
	ck(Saturday, "Saturday")
	ck(Sunday, "Sunday")
	ck(-127, "Day(-127)")
	ck(127, "Day(127)")
}

func ck(day Day, str string) {
	if fmt.Sprint(day) != str {
		panic("day.go: " + str)
	}
}
