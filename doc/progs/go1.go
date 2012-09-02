// compile
// this file will output a list of filenames in cwd, not suitable for cmpout

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains examples to embed in the Go 1 release notes document.

package main

import (
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"testing"
	"time"
	"unicode"
)

func main() {
	flag.Parse()
	stringAppend()
	mapDelete()
	mapIteration()
	multipleAssignment()
	structEquality()
	compositeLiterals()
	runeType()
	errorExample()
	timePackage()
	walkExample()
	osIsExist()
}

var timeout = flag.Duration("timeout", 30*time.Second, "how long to wait for completion")

func init() {
	// canonicalize the logging
	log.SetFlags(0)
}

func mapDelete() {
	m := map[string]int{"7": 7, "23": 23}
	k := "7"
	delete(m, k)
	if m["7"] != 0 || m["23"] != 23 {
		log.Fatal("mapDelete:", m)
	}
}

func stringAppend() {
	greeting := []byte{}
	greeting = append(greeting, []byte("hello ")...)
	greeting = append(greeting, "world"...)
	if string(greeting) != "hello world" {
		log.Fatal("stringAppend: ", string(greeting))
	}
}

func mapIteration() {
	m := map[string]int{"Sunday": 0, "Monday": 1}
	for name, value := range m {
		// This loop should not assume Sunday will be visited first.
		f(name, value)
	}
}

func f(string, int) {
}

func assert(t bool) {
	if !t {
		log.Panic("assertion fail")
	}
}

func multipleAssignment() {
	sa := []int{1, 2, 3}
	i := 0
	i, sa[i] = 1, 2 // sets i = 1, sa[0] = 2

	sb := []int{1, 2, 3}
	j := 0
	sb[j], j = 2, 1 // sets sb[0] = 2, j = 1

	sc := []int{1, 2, 3}
	sc[0], sc[0] = 1, 2 // sets sc[0] = 1, then sc[0] = 2 (so sc[0] = 2 at end)

	assert(i == 1 && sa[0] == 2)
	assert(j == 1 && sb[0] == 2)
	assert(sc[0] == 2)
}

func structEquality() {
	type Day struct {
		long  string
		short string
	}
	Christmas := Day{"Christmas", "XMas"}
	Thanksgiving := Day{"Thanksgiving", "Turkey"}
	holiday := map[Day]bool{
		Christmas:    true,
		Thanksgiving: true,
	}
	fmt.Printf("Christmas is a holiday: %t\n", holiday[Christmas])
}

func compositeLiterals() {
	type Date struct {
		month string
		day   int
	}
	// Struct values, fully qualified; always legal.
	holiday1 := []Date{
		Date{"Feb", 14},
		Date{"Nov", 11},
		Date{"Dec", 25},
	}
	// Struct values, type name elided; always legal.
	holiday2 := []Date{
		{"Feb", 14},
		{"Nov", 11},
		{"Dec", 25},
	}
	// Pointers, fully qualified, always legal.
	holiday3 := []*Date{
		&Date{"Feb", 14},
		&Date{"Nov", 11},
		&Date{"Dec", 25},
	}
	// Pointers, type name elided; legal in Go 1.
	holiday4 := []*Date{
		{"Feb", 14},
		{"Nov", 11},
		{"Dec", 25},
	}
	// STOP OMIT
	_, _, _, _ = holiday1, holiday2, holiday3, holiday4
}

func runeType() {
	// STARTRUNE OMIT
	delta := 'δ' // delta has type rune.
	var DELTA rune
	DELTA = unicode.ToUpper(delta)
	epsilon := unicode.ToLower(DELTA + 1)
	if epsilon != 'δ'+1 {
		log.Fatal("inconsistent casing for Greek")
	}
	// ENDRUNE OMIT
}

// START ERROR EXAMPLE OMIT
type SyntaxError struct {
	File    string
	Line    int
	Message string
}

func (se *SyntaxError) Error() string {
	return fmt.Sprintf("%s:%d: %s", se.File, se.Line, se.Message)
}

// END ERROR EXAMPLE OMIT

func errorExample() {
	var ErrSyntax = errors.New("syntax error")
	_ = ErrSyntax
	se := &SyntaxError{"file", 7, "error"}
	got := fmt.Sprint(se)
	const expect = "file:7: error"
	if got != expect {
		log.Fatalf("errorsPackage: expected %q got %q", expect, got)
	}
}

// sleepUntil sleeps until the specified time. It returns immediately if it's too late.
func sleepUntil(wakeup time.Time) {
	now := time.Now() // A Time.
	if !wakeup.After(now) {
		return
	}
	delta := wakeup.Sub(now) // A Duration.
	fmt.Printf("Sleeping for %.3fs\n", delta.Seconds())
	time.Sleep(delta)
}

func timePackage() {
	sleepUntil(time.Now().Add(123 * time.Millisecond))
}

func walkExample() {
	// STARTWALK OMIT
	markFn := func(path string, info os.FileInfo, err error) error {
		if path == "pictures" { // Will skip walking of directory pictures and its contents.
			return filepath.SkipDir
		}
		if err != nil {
			return err
		}
		log.Println(path)
		return nil
	}
	err := filepath.Walk(".", markFn)
	if err != nil {
		log.Fatal(err)
	}
	// ENDWALK OMIT
}

func initializationFunction(c chan int) {
	c <- 1
}

var PackageGlobal int

func init() {
	c := make(chan int)
	go initializationFunction(c)
	PackageGlobal = <-c
}

func BenchmarkSprintf(b *testing.B) {
	// Verify correctness before running benchmark.
	b.StopTimer()
	got := fmt.Sprintf("%x", 23)
	const expect = "17"
	if expect != got {
		b.Fatalf("expected %q; got %q", expect, got)
	}
	b.StartTimer()
	for i := 0; i < b.N; i++ {
		fmt.Sprintf("%x", 23)
	}
}

func osIsExist() {
	name := "go1.go"
	f, err := os.OpenFile(name, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0600)
	if os.IsExist(err) {
		log.Printf("%s already exists", name)
	}
	_ = f
}
