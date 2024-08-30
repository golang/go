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

	// fmt.Println uses these arguments to print floats
	fmt64 := strconv.FormatFloat(v, 'g', -1, 64)
	fmt.Printf("%T, %v\n", fmt64, fmt64)

	// Output:
	// string, 3.1415927E+00
	// string, 3.1415926535E+00
	// string, 3.1415926535
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

func ExampleIsGraphic() {
	shamrock := strconv.IsGraphic('☘')
	fmt.Println(shamrock)

	a := strconv.IsGraphic('a')
	fmt.Println(a)

	bel := strconv.IsGraphic('\007')
	fmt.Println(bel)

	// Output:
	// true
	// true
	// false
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
	if s, err := strconv.ParseFloat("NaN", 32); err == nil {
		fmt.Printf("%T, %v\n", s, s)
	}
	// ParseFloat is case insensitive
	if s, err := strconv.ParseFloat("nan", 32); err == nil {
		fmt.Printf("%T, %v\n", s, s)
	}
	if s, err := strconv.ParseFloat("inf", 32); err == nil {
		fmt.Printf("%T, %v\n", s, s)
	}
	if s, err := strconv.ParseFloat("+Inf", 32); err == nil {
		fmt.Printf("%T, %v\n", s, s)
	}
	if s, err := strconv.ParseFloat("-Inf", 32); err == nil {
		fmt.Printf("%T, %v\n", s, s)
	}
	if s, err := strconv.ParseFloat("-0", 32); err == nil {
		fmt.Printf("%T, %v\n", s, s)
	}
	if s, err := strconv.ParseFloat("+0", 32); err == nil {
		fmt.Printf("%T, %v\n", s, s)
	}

	// Output:
	// float64, 3.1415927410125732
	// float64, 3.1415926535
	// float64, NaN
	// float64, NaN
	// float64, +Inf
	// float64, +Inf
	// float64, -Inf
	// float64, -0
	// float64, 0
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
	// This string literal contains a tab character.
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

func ExampleQuoteRuneToGraphic() {
	s := strconv.QuoteRuneToGraphic('☺')
	fmt.Println(s)

	s = strconv.QuoteRuneToGraphic('\u263a')
	fmt.Println(s)

	s = strconv.QuoteRuneToGraphic('\u000a')
	fmt.Println(s)

	s = strconv.QuoteRuneToGraphic('	') // tab character
	fmt.Println(s)

	// Output:
	// '☺'
	// '☺'
	// '\n'
	// '\t'
}

func ExampleQuoteToASCII() {
	// This string literal contains a tab character.
	s := strconv.QuoteToASCII(`"Fran & Freddie's Diner	☺"`)
	fmt.Println(s)

	// Output:
	// "\"Fran & Freddie's Diner\t\u263a\""
}

func ExampleQuoteToGraphic() {
	s := strconv.QuoteToGraphic("☺")
	fmt.Println(s)

	// This string literal contains a tab character.
	s = strconv.QuoteToGraphic("This is a \u263a	\u000a")
	fmt.Println(s)

	s = strconv.QuoteToGraphic(`" This is a ☺ \n "`)
	fmt.Println(s)

	// Output:
	// "☺"
	// "This is a ☺\t\n"
	// "\" This is a ☺ \\n \""
}

func ExampleQuotedPrefix() {
	s, err := strconv.QuotedPrefix("not a quoted string")
	fmt.Printf("%q, %v\n", s, err)
	s, err = strconv.QuotedPrefix("\"double-quoted string\" with trailing text")
	fmt.Printf("%q, %v\n", s, err)
	s, err = strconv.QuotedPrefix("`or backquoted` with more trailing text")
	fmt.Printf("%q, %v\n", s, err)
	s, err = strconv.QuotedPrefix("'\u263a' is also okay")
	fmt.Printf("%q, %v\n", s, err)

	// Output:
	// "", invalid syntax
	// "\"double-quoted string\"", <nil>
	// "`or backquoted`", <nil>
	// "'☺'", <nil>
}

func ExampleUnquote() {
	s, err := strconv.Unquote("You can't unquote a string without quotes")
	fmt.Printf("%q, %v\n", s, err)
	s, err = strconv.Unquote("\"The string must be either double-quoted\"")
	fmt.Printf("%q, %v\n", s, err)
	s, err = strconv.Unquote("`or backquoted.`")
	fmt.Printf("%q, %v\n", s, err)
	s, err = strconv.Unquote("'\u263a'") // single character only allowed in single quotes
	fmt.Printf("%q, %v\n", s, err)
	s, err = strconv.Unquote("'\u2639\u2639'")
	fmt.Printf("%q, %v\n", s, err)

	// Output:
	// "", invalid syntax
	// "The string must be either double-quoted", <nil>
	// "or backquoted.", <nil>
	// "☺", <nil>
	// "", invalid syntax
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
