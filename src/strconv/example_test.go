// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	"fmt"
	"log"
	"strconv"
)

func ExampleAppendBool() {
	b := []byte("bool:")
	b = strconv.AppendBool(b, true)
	fmt.Println(string(b))

	// Output:
	// bool:true
}

func ExampleAppendFloat() {
	b32 := []byte("float32:")
	b32 = strconv.AppendFloat(b32, 3.1415926535, 'E', -1, 32)
	fmt.Println(string(b32))

	b64 := []byte("float64:")
	b64 = strconv.AppendFloat(b64, 3.1415926535, 'E', -1, 64)
	fmt.Println(string(b64))

	// Output:
	// float32:3.1415927E+00
	// float64:3.1415926535E+00
}

func ExampleAppendInt() {
	b10 := []byte("int (base 10):")
	b10 = strconv.AppendInt(b10, -42, 10)
	fmt.Println(string(b10))

	b16 := []byte("int (base 16):")
	b16 = strconv.AppendInt(b16, -42, 16)
	fmt.Println(string(b16))

	// Output:
	// int (base 10):-42
	// int (base 16):-2a
}

func ExampleAppendQuote() {
	b := []byte("quote:")
	b = strconv.AppendQuote(b, `"Fran & Freddie's Diner"`)
	fmt.Println(string(b))

	// Output:
	// quote:"\"Fran & Freddie's Diner\""
}

func ExampleAppendQuoteRune() {
	b := []byte("rune:")
	b = strconv.AppendQuoteRune(b, '☺')
	fmt.Println(string(b))

	// Output:
	// rune:'☺'
}

func ExampleAppendQuoteRuneToASCII() {
	b := []byte("rune (ascii):")
	b = strconv.AppendQuoteRuneToASCII(b, '☺')
	fmt.Println(string(b))

	// Output:
	// rune (ascii):'\u263a'
}

func ExampleAppendQuoteToASCII() {
	b := []byte("quote (ascii):")
	b = strconv.AppendQuoteToASCII(b, `"Fran & Freddie's Diner"`)
	fmt.Println(string(b))

	// Output:
	// quote (ascii):"\"Fran & Freddie's Diner\""
}

func ExampleAppendUint() {
	b10 := []byte("uint (base 10):")
	b10 = strconv.AppendUint(b10, 42, 10)
	fmt.Println(string(b10))

	b16 := []byte("uint (base 16):")
	b16 = strconv.AppendUint(b16, 42, 16)
	fmt.Println(string(b16))

	// Output:
	// uint (base 10):42
	// uint (base 16):2a
}

func ExampleAtoi() {
	v := "10"
	if s, err := strconv.Atoi(v); err == nil {
		fmt.Printf("%T, %v", s, s)
	}

	// Output:
	// int, 10
}

func ExampleCanBackquote() {
	fmt.Println(strconv.CanBackquote("Fran & Freddie's Diner ☺"))
	fmt.Println(strconv.CanBackquote("`can't backquote this`"))

	// Output:
	// true
	// false
}

func ExampleFormatBool() {
	v := true
	s := strconv.FormatBool(v)
	fmt.Printf("%T, %v\n", s, s)

	// Output:
	// string, true
}

func ExampleFormatFloat() {
	v := 3.1415926535

	s32 := strconv.FormatFloat(v, 'E', -1, 32)
	fmt.Printf("%T, %v\n", s32, s32)

	s64 := strconv.FormatFloat(v, 'E', -1, 64)
	fmt.Printf("%T, %v\n", s64, s64)

	// Output:
	// string, 3.1415927E+00
	// string, 3.1415926535E+00
}

func ExampleFormatInt() {
	v := int64(-42)

	s10 := strconv.FormatInt(v, 10)
	fmt.Printf("%T, %v\n", s10, s10)

	s16 := strconv.FormatInt(v, 16)
	fmt.Printf("%T, %v\n", s16, s16)

	// Output:
	// string, -42
	// string, -2a
}

func ExampleFormatUint() {
	v := uint64(42)

	s10 := strconv.FormatUint(v, 10)
	fmt.Printf("%T, %v\n", s10, s10)

	s16 := strconv.FormatUint(v, 16)
	fmt.Printf("%T, %v\n", s16, s16)

	// Output:
	// string, 42
	// string, 2a
}

func ExampleIsPrint() {
	c := strconv.IsPrint('\u263a')
	fmt.Println(c)

	bel := strconv.IsPrint('\007')
	fmt.Println(bel)

	// Output:
	// true
	// false
}

func ExampleItoa() {
	i := 10
	s := strconv.Itoa(i)
	fmt.Printf("%T, %v\n", s, s)

	// Output:
	// string, 10
}

func ExampleParseBool() {
	v := "true"
	if s, err := strconv.ParseBool(v); err == nil {
		fmt.Printf("%T, %v\n", s, s)
	}

	// Output:
	// bool, true
}

func ExampleParseFloat() {
	v := "3.1415926535"
	if s, err := strconv.ParseFloat(v, 32); err == nil {
		fmt.Printf("%T, %v\n", s, s)
	}
	if s, err := strconv.ParseFloat(v, 64); err == nil {
		fmt.Printf("%T, %v\n", s, s)
	}

	// Output:
	// float64, 3.1415927410125732
	// float64, 3.1415926535
}

func ExampleParseInt() {
	v32 := "-354634382"
	if s, err := strconv.ParseInt(v32, 10, 32); err == nil {
		fmt.Printf("%T, %v\n", s, s)
	}
	if s, err := strconv.ParseInt(v32, 16, 32); err == nil {
		fmt.Printf("%T, %v\n", s, s)
	}

	v64 := "-3546343826724305832"
	if s, err := strconv.ParseInt(v64, 10, 64); err == nil {
		fmt.Printf("%T, %v\n", s, s)
	}
	if s, err := strconv.ParseInt(v64, 16, 64); err == nil {
		fmt.Printf("%T, %v\n", s, s)
	}

	// Output:
	// int64, -354634382
	// int64, -3546343826724305832
}

func ExampleParseUint() {
	v := "42"
	if s, err := strconv.ParseUint(v, 10, 32); err == nil {
		fmt.Printf("%T, %v\n", s, s)
	}
	if s, err := strconv.ParseUint(v, 10, 64); err == nil {
		fmt.Printf("%T, %v\n", s, s)
	}

	// Output:
	// uint64, 42
	// uint64, 42
}

func ExampleQuote() {
	s := strconv.Quote(`"Fran & Freddie's Diner	☺"`)
	fmt.Println(s)

	// Output:
	// "\"Fran & Freddie's Diner\t☺\""
}

func ExampleQuoteRune() {
	s := strconv.QuoteRune('☺')
	fmt.Println(s)

	// Output:
	// '☺'
}

func ExampleQuoteRuneToASCII() {
	s := strconv.QuoteRuneToASCII('☺')
	fmt.Println(s)

	// Output:
	// '\u263a'
}

func ExampleQuoteToASCII() {
	s := strconv.QuoteToASCII(`"Fran & Freddie's Diner	☺"`)
	fmt.Println(s)

	// Output:
	// "\"Fran & Freddie's Diner\t\u263a\""
}

func ExampleUnquote() {
	test := func(s string) {
		t, err := strconv.Unquote(s)
		if err != nil {
			fmt.Printf("Unquote(%#v): %v\n", s, err)
		} else {
			fmt.Printf("Unquote(%#v) = %v\n", s, t)
		}
	}

	s := `\"Fran & Freddie's Diner\t\u263a\"\"`
	// If the string doesn't have quotes, it can't be unquoted.
	test(s) // invalid syntax
	test("`" + s + "`")
	test(`"` + s + `"`)
	test(`'\u263a'`)

	// Output:
	// Unquote("\\\"Fran & Freddie's Diner\\t\\u263a\\\"\\\""): invalid syntax
	// Unquote("`\\\"Fran & Freddie's Diner\\t\\u263a\\\"\\\"`") = \"Fran & Freddie's Diner\t\u263a\"\"
	// Unquote("\"\\\"Fran & Freddie's Diner\\t\\u263a\\\"\\\"\"") = "Fran & Freddie's Diner	☺""
	// Unquote("'\\u263a'") = ☺
}

func ExampleUnquoteChar() {
	v, mb, t, err := strconv.UnquoteChar(`\"Fran & Freddie's Diner\"`, '"')
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("value:", string(v))
	fmt.Println("multibyte:", mb)
	fmt.Println("tail:", t)

	// Output:
	// value: "
	// multibyte: false
	// tail: Fran & Freddie's Diner\"
}

func ExampleNumError() {
	str := "Not a number"
	if _, err := strconv.ParseFloat(str, 64); err != nil {
		e := err.(*strconv.NumError)
		fmt.Println("Func:", e.Func)
		fmt.Println("Num:", e.Num)
		fmt.Println("Err:", e.Err)
		fmt.Println(err)
	}

	// Output:
	// Func: ParseFloat
	// Num: Not a number
	// Err: invalid syntax
	// strconv.ParseFloat: parsing "Not a number": invalid syntax
}
