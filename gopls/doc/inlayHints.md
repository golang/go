# Hints

This document describes the inlay hints that `gopls` uses inside the editor.

<!-- BEGIN Hints: DO NOT MANUALLY EDIT THIS SECTION -->
## **assign_variable_types**

Enable/disable inlay hints for variable types in assign statements:

	i/* int/*, j/* int/* := 0, len(r)-1

**Disabled by default. Enable it by setting `"hints": {"assign_variable_types": true}`.**

## **composite_literal_fields**

Enable/disable inlay hints for composite literal field names:

	{in: "Hello, world", want: "dlrow ,olleH"}

**Disabled by default. Enable it by setting `"hints": {"composite_literal_fields": true}`.**

## **composite_literal_types**

Enable/disable inlay hints for composite literal types:

	for _, c := range []struct {
		in, want string
	}{
		/*struct{ in string; want string }*/{"Hello, world", "dlrow ,olleH"},
	}

**Disabled by default. Enable it by setting `"hints": {"composite_literal_types": true}`.**

## **constant_values**

Enable/disable inlay hints for constant values:

	const (
		KindNone   Kind = iota/* = 0*/
		KindPrint/*  = 1*/
		KindPrintf/* = 2*/
		KindErrorf/* = 3*/
	)

**Disabled by default. Enable it by setting `"hints": {"constant_values": true}`.**

## **function_type_parameters**

Enable/disable inlay hints for implicit type parameters on generic functions:

	myFoo/*[int, string]*/(1, "hello")

**Disabled by default. Enable it by setting `"hints": {"function_type_parameters": true}`.**

## **parameter_names**

Enable/disable inlay hints for parameter names:

	parseInt(/* str: */ "123", /* radix: */ 8)

**Disabled by default. Enable it by setting `"hints": {"parameter_names": true}`.**

## **range_variable_types**

Enable/disable inlay hints for variable types in range statements:

	for k/* int*/, v/* string/* := range []string{} {
		fmt.Println(k, v)
	}

**Disabled by default. Enable it by setting `"hints": {"range_variable_types": true}`.**

<!-- END Hints: DO NOT MANUALLY EDIT THIS SECTION -->
