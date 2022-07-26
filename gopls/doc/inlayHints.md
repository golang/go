# Hints

This document describes the inlay hints that `gopls` uses inside the editor.

<!-- BEGIN Hints: DO NOT MANUALLY EDIT THIS SECTION -->
## **assignVariableTypes**

Enable/disable inlay hints for variable types in assign statements:
```go
	i/* int*/, j/* int*/ := 0, len(r)-1
```

**Disabled by default. Enable it by setting `"hints": {"assignVariableTypes": true}`.**

## **compositeLiteralFields**

Enable/disable inlay hints for composite literal field names:
```go
	{/*in: */"Hello, world", /*want: */"dlrow ,olleH"}
```

**Disabled by default. Enable it by setting `"hints": {"compositeLiteralFields": true}`.**

## **compositeLiteralTypes**

Enable/disable inlay hints for composite literal types:
```go
	for _, c := range []struct {
		in, want string
	}{
		/*struct{ in string; want string }*/{"Hello, world", "dlrow ,olleH"},
	}
```

**Disabled by default. Enable it by setting `"hints": {"compositeLiteralTypes": true}`.**

## **constantValues**

Enable/disable inlay hints for constant values:
```go
	const (
		KindNone   Kind = iota/* = 0*/
		KindPrint/*  = 1*/
		KindPrintf/* = 2*/
		KindErrorf/* = 3*/
	)
```

**Disabled by default. Enable it by setting `"hints": {"constantValues": true}`.**

## **functionTypeParameters**

Enable/disable inlay hints for implicit type parameters on generic functions:
```go
	myFoo/*[int, string]*/(1, "hello")
```

**Disabled by default. Enable it by setting `"hints": {"functionTypeParameters": true}`.**

## **parameterNames**

Enable/disable inlay hints for parameter names:
```go
	parseInt(/* str: */ "123", /* radix: */ 8)
```

**Disabled by default. Enable it by setting `"hints": {"parameterNames": true}`.**

## **rangeVariableTypes**

Enable/disable inlay hints for variable types in range statements:
```go
	for k/* int*/, v/* string*/ := range []string{} {
		fmt.Println(k, v)
	}
```

**Disabled by default. Enable it by setting `"hints": {"rangeVariableTypes": true}`.**

<!-- END Hints: DO NOT MANUALLY EDIT THIS SECTION -->
