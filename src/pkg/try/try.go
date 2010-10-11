// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package try contains the executable part of the gotry command.
// It is not intended for general use.
package try

import (
	"fmt"
	"io"
	"os"
	"reflect"
	"unicode"
)

var output io.Writer = os.Stdout // redirected when testing

// Main is called directly from the gotry-generated Go source file to perform
// the evaluations.
func Main(pkg, firstArg string, functions map[string]interface{}, args []interface{}) {
	switch len(args) {
	case 0:
		// Nothing to do.
	case 1:
		// Compiler has already evaluated the expression; just print the result.
		printSlice(firstArg, args)
	default:
		// See if methods satisfy the expressions.
		tryMethods(pkg, firstArg, args)
		// See if functions satisfy the expressions.
		for name, fn := range functions {
			tryFunction(pkg, name, fn, args)
		}
	}
}

// printSlice prints the zeroth element of the args slice, which should (by construction)
// itself be a slice of interface{}.
func printSlice(firstArg string, args []interface{}) {
	// Args should be length 1 and a slice.
	if len(args) != 1 {
		return
	}
	arg, ok := args[0].([]interface{})
	if !ok {
		return
	}
	fmt.Fprintf(output, "%s = ", firstArg)
	if len(arg) > 1 {
		fmt.Fprint(output, "(")
	}
	for i, a := range arg {
		if i > 0 {
			fmt.Fprint(output, ", ")
		}
		fmt.Fprintf(output, "%#v", a)
	}
	if len(arg) > 1 {
		fmt.Fprint(output, ")")
	}
	fmt.Fprint(output, "\n")
}

// tryMethods sees if the zeroth arg has methods, and if so treats them as potential
// functions to satisfy the remaining arguments.
func tryMethods(pkg, firstArg string, args []interface{}) {
	defer func() { recover() }()
	// Is the first argument something with methods?
	v := reflect.NewValue(args[0])
	typ := v.Type()
	if typ.NumMethod() == 0 {
		return
	}
	for i := 0; i < typ.NumMethod(); i++ {
		if unicode.IsUpper(int(typ.Method(i).Name[0])) {
			tryMethod(pkg, firstArg, typ.Method(i), args)
		}
	}
}

// tryMethod converts a method to a function for tryOneFunction.
func tryMethod(pkg, firstArg string, method reflect.Method, args []interface{}) {
	rfn := method.Func
	typ := method.Type
	name := method.Name
	tryOneFunction(pkg, firstArg, name, typ, rfn, args)
}

// tryFunction sees if fn satisfies the arguments.
func tryFunction(pkg, name string, fn interface{}, args []interface{}) {
	defer func() { recover() }()
	rfn := reflect.NewValue(fn).(*reflect.FuncValue)
	typ := rfn.Type().(*reflect.FuncType)
	tryOneFunction(pkg, "", name, typ, rfn, args)
}

// tryOneFunction is the common code for tryMethod and tryFunction.
func tryOneFunction(pkg, firstArg, name string, typ *reflect.FuncType, rfn *reflect.FuncValue, args []interface{}) {
	// Any results?
	if typ.NumOut() == 0 {
		return // Nothing to do.
	}
	// Right number of arguments + results?
	if typ.NumIn()+typ.NumOut() != len(args) {
		return
	}
	// Right argument and result types?
	for i, a := range args {
		if i < typ.NumIn() {
			if !compatible(a, typ.In(i)) {
				return
			}
		} else {
			if !compatible(a, typ.Out(i-typ.NumIn())) {
				return
			}
		}
	}
	// Build the call args.
	argsVal := make([]reflect.Value, typ.NumIn()+typ.NumOut())
	for i, a := range args {
		argsVal[i] = reflect.NewValue(a)
	}
	// Call the function and see if the results are as expected.
	resultVal := rfn.Call(argsVal[:typ.NumIn()])
	for i, v := range resultVal {
		if !reflect.DeepEqual(v.Interface(), args[i+typ.NumIn()]) {
			return
		}
	}
	// Present the result including a godoc command to get more information.
	firstIndex := 0
	if firstArg != "" {
		fmt.Fprintf(output, "%s.%s(", firstArg, name)
		firstIndex = 1
	} else {
		fmt.Fprintf(output, "%s.%s(", pkg, name)
	}
	for i := firstIndex; i < typ.NumIn(); i++ {
		if i > firstIndex {
			fmt.Fprint(output, ", ")
		}
		fmt.Fprintf(output, "%#v", args[i])
	}
	fmt.Fprint(output, ") = ")
	if typ.NumOut() > 1 {
		fmt.Fprint(output, "(")
	}
	for i := 0; i < typ.NumOut(); i++ {
		if i > 0 {
			fmt.Fprint(output, ", ")
		}
		fmt.Fprintf(output, "%#v", resultVal[i].Interface())
	}
	if typ.NumOut() > 1 {
		fmt.Fprint(output, ")")
	}
	fmt.Fprintf(output, "  // godoc %s %s\n", pkg, name)
}

// compatible reports whether the argument is compatible with the type.
func compatible(arg interface{}, typ reflect.Type) bool {
	if reflect.Typeof(arg) == typ {
		return true
	}
	if arg == nil {
		// nil is OK if the type is an interface.
		if _, ok := typ.(*reflect.InterfaceType); ok {
			return true
		}
	}
	return false
}
