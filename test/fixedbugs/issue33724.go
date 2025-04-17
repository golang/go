// run
package main

import (
	"fmt"
	"runtime/debug"
	"strings"
)

type Inner struct {
	Err int
}

func (i *Inner) NotExpectedInStackTrace() int {
	if i == nil {
		return 86
	}
	return 17 + i.Err
}

type Outer struct {
	Inner
}

func ExpectedInStackTrace() {
	var o *Outer
	println(o.NotExpectedInStackTrace())
}

func main() {
    defer func() {
        if r := recover(); r != nil {
        	stacktrace := string(debug.Stack())
        	if strings.Contains(stacktrace, "NotExpectedInStackTrace") {
        		fmt.Println("FAIL, stacktrace contains NotExpectedInStackTrace")
        	}
        	if !strings.Contains(stacktrace, "ExpectedInStackTrace") {
        		fmt.Println("FAIL, stacktrace does not contain ExpectedInStackTrace")
        	}
        } else {
        	fmt.Println("FAIL, should have panicked but did not")
        }
    }()
    ExpectedInStackTrace()
}
