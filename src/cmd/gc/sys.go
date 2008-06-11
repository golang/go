// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


package foop	// rename to avoid redeclaration

func	mal(uint32) *byte;
func	breakpoint();
func	panicl(int32);

func	printbool(bool);
func	printfloat(double);
func	printint(int64);
func	printstring(string);
func	printpointer(*byte);

func	catstring(string, string) string;
func	cmpstring(string, string) int32;
func	slicestring(string, int32, int32) string;
func	indexstring(string, int32) byte;
func	intstring(int64) string;
func	byteastring(*byte, int32) string;
func	mkiface(*byte, *byte, *struct{}) interface{};

func	frexp(float64) (int32, float64);	// break fp into exp,fract
func	ldexp(int32, float64) float64;		// make fp from exp,fract
func	modf(float64) (float64, float64);	// break fp into double.double

export
	mal
	breakpoint
	panicl

	printbool
	printfloat
	printint
	printstring
	printpointer

	catstring
	cmpstring
	slicestring
	indexstring
	intstring
	byteastring
	mkiface

	frexp
	ldexp
	modf
	;
