package main

import "cmd/cgo/internal/testsanitizers/testdata/asan_linkerx/p"

func pstring(s *string) {
	println(*s)
}

func main() {
	all := []*string{
		&S1, &S2, &S3, &S4, &S5, &S6, &S7, &S8, &S9, &S10,
		&p.S1, &p.S2, &p.S3, &p.S4, &p.S5, &p.S6, &p.S7, &p.S8, &p.S9, &p.S10,
	}
	for _, ps := range all {
		pstring(ps)
	}
}

var S1 string
var S2 string
var S3 string
var S4 string
var S5 string
var S6 string
var S7 string
var S8 string
var S9 string
var S10 string
