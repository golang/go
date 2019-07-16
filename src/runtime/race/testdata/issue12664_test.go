// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package race_test

import (
	"fmt"
	"testing"
)

var issue12664 = "hi"

func TestRaceIssue12664(t *testing.T) {
	c := make(chan struct{})
	go func() {
		issue12664 = "bye"
		close(c)
	}()
	fmt.Println(issue12664)
	<-c
}

type MyI interface {
	foo()
}

type MyT int

func (MyT) foo() {
}

var issue12664_2 MyT = 0

func TestRaceIssue12664_2(t *testing.T) {
	c := make(chan struct{})
	go func() {
		issue12664_2 = 1
		close(c)
	}()
	func(x MyI) {
		// Never true, but prevents inlining.
		if x.(MyT) == -1 {
			close(c)
		}
	}(issue12664_2)
	<-c
}

var issue12664_3 MyT = 0

func TestRaceIssue12664_3(t *testing.T) {
	c := make(chan struct{})
	go func() {
		issue12664_3 = 1
		close(c)
	}()
	var r MyT
	var i interface{} = r
	issue12664_3 = i.(MyT)
	<-c
}

var issue12664_4 MyT = 0

func TestRaceIssue12664_4(t *testing.T) {
	c := make(chan struct{})
	go func() {
		issue12664_4 = 1
		close(c)
	}()
	var r MyT
	var i MyI = r
	issue12664_4 = i.(MyT)
	<-c
}
