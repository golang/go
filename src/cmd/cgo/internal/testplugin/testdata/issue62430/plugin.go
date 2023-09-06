package main

import (
	"unicode"
)

func F(s string) *unicode.RangeTable {
	return unicode.Categories[s]
}

func main() {}
