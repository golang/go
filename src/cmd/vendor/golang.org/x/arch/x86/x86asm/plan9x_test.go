// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86asm

import (
	"strings"
	"testing"
)

func TestPlan932Manual(t *testing.T)   { testPlan932(t, hexCases(t, plan9ManualTests)) }
func TestPlan932Testdata(t *testing.T) { testPlan932(t, concat(basicPrefixes, testdataCases(t))) }
func TestPlan932ModRM(t *testing.T)    { testPlan932(t, concat(basicPrefixes, enumModRM)) }
func TestPlan932OneByte(t *testing.T)  { testBasic(t, testPlan932) }
func TestPlan9320F(t *testing.T)       { testBasic(t, testPlan932, 0x0F) }
func TestPlan9320F38(t *testing.T)     { testBasic(t, testPlan932, 0x0F, 0x38) }
func TestPlan9320F3A(t *testing.T)     { testBasic(t, testPlan932, 0x0F, 0x3A) }
func TestPlan932Prefix(t *testing.T)   { testPrefix(t, testPlan932) }

func TestPlan964Manual(t *testing.T)   { testPlan964(t, hexCases(t, plan9ManualTests)) }
func TestPlan964Testdata(t *testing.T) { testPlan964(t, concat(basicPrefixes, testdataCases(t))) }
func TestPlan964ModRM(t *testing.T)    { testPlan964(t, concat(basicPrefixes, enumModRM)) }
func TestPlan964OneByte(t *testing.T)  { testBasic(t, testPlan964) }
func TestPlan9640F(t *testing.T)       { testBasic(t, testPlan964, 0x0F) }
func TestPlan9640F38(t *testing.T)     { testBasic(t, testPlan964, 0x0F, 0x38) }
func TestPlan9640F3A(t *testing.T)     { testBasic(t, testPlan964, 0x0F, 0x3A) }
func TestPlan964Prefix(t *testing.T)   { testPrefix(t, testPlan964) }

func TestPlan964REXTestdata(t *testing.T) {
	testPlan964(t, filter(concat3(basicPrefixes, rexPrefixes, testdataCases(t)), isValidREX))
}
func TestPlan964REXModRM(t *testing.T)   { testPlan964(t, concat3(basicPrefixes, rexPrefixes, enumModRM)) }
func TestPlan964REXOneByte(t *testing.T) { testBasicREX(t, testPlan964) }
func TestPlan964REX0F(t *testing.T)      { testBasicREX(t, testPlan964, 0x0F) }
func TestPlan964REX0F38(t *testing.T)    { testBasicREX(t, testPlan964, 0x0F, 0x38) }
func TestPlan964REX0F3A(t *testing.T)    { testBasicREX(t, testPlan964, 0x0F, 0x3A) }
func TestPlan964REXPrefix(t *testing.T)  { testPrefixREX(t, testPlan964) }

// plan9ManualTests holds test cases that will be run by TestPlan9Manual32 and TestPlan9Manual64.
// If you are debugging a few cases that turned up in a longer run, it can be useful
// to list them here and then use -run=Plan9Manual, particularly with tracing enabled.
var plan9ManualTests = `
`

// allowedMismatchPlan9 reports whether the mismatch between text and dec
// should be allowed by the test.
func allowedMismatchPlan9(text string, size int, inst *Inst, dec ExtInst) bool {
	return false
}

// Instructions known to us but not to plan9.
var plan9Unsupported = strings.Fields(`
`)
