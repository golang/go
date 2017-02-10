// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package demangle

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"
	"testing"
)

var verbose = flag.Bool("verbose", false, "print each demangle-expected symbol")

const filename = "testdata/demangle-expected"

// A list of exceptions from demangle-expected that we do not handle
// the same as the standard demangler.  We keep a list of exceptions
// so that we can use an exact copy of the file.  These exceptions are
// all based on different handling of a substitution that refers to a
// template parameter.  The standard demangler seems to have a bug in
// which template it uses when a reference or rvalue-reference refers
// to a substitution that resolves to a template parameter.
var exceptions = map[string]bool{
	"_ZSt7forwardIRN1x14refobjiteratorINS0_3refINS0_4mime30multipart_section_processorObjIZ15get_body_parserIZZN14mime_processor21make_section_iteratorERKNS2_INS3_10sectionObjENS0_10ptrrefBaseEEEbENKUlvE_clEvEUlSB_bE_ZZNS6_21make_section_iteratorESB_bENKSC_clEvEUlSB_E0_ENS1_INS2_INS0_20outputrefiteratorObjIiEES8_EEEERKSsSB_OT_OT0_EUlmE_NS3_32make_multipart_default_discarderISP_EEEES8_EEEEEOT_RNSt16remove_referenceISW_E4typeE": true,
	"_ZN3mdr16in_cached_threadIRZNK4cudr6GPUSet17parallel_for_eachIZN5tns3d20shape_representation7compute7GPUImpl7executeERKNS_1AINS_7ptr_refIKjEELl3ELl3ENS_8c_strideILl1ELl0EEEEERKNS8_INS9_IjEELl4ELl1ESD_EEEUliRKNS1_7ContextERNS7_5StateEE_JSt6vectorISO_SaISO_EEEEEvOT_DpRT0_EUlSP_E_JSt17reference_wrapperISO_EEEENS_12ScopedFutureIDTclfp_spcl7forwardISW_Efp0_EEEEESV_DpOSW_":                                                        true,
	"_ZNSt9_Any_data9_M_accessIPZN3sel8Selector6SetObjI3FooJPKcMS4_FviEEEEvRT_DpT0_EUlvE_EESA_v":                                                                                                                   true,
	"_ZNSt9_Any_data9_M_accessIPZN13ThreadManager7newTaskIRSt5_BindIFSt7_Mem_fnIM5DiaryFivEEPS5_EEIEEESt6futureINSt9result_ofIFT_DpT0_EE4typeEEOSF_DpOSG_EUlvE_EERSF_v":                                            true,
	"_ZNSt9_Any_data9_M_accessIPZN6cereal18polymorphic_detail15getInputBindingINS1_16JSONInputArchiveEEENS1_6detail15InputBindingMapIT_E11SerializersERS7_jEUlPvRSt10unique_ptrIvNS5_12EmptyDeleterIvEEEE0_EESA_v": true,
	"_ZNSt9_Any_data9_M_accessIPZ4postISt8functionIFvvEEEvOT_EUlvE_EERS5_v":                                                                                                                                        true,
	"_ZNSt9_Any_data9_M_accessIPZN13ThreadManager10futureTaskISt5_BindIFSt7_Mem_fnIM6RunnerFvvEEPS5_EEEEvOT_EUlvE_EERSC_v":                                                                                         true,
}

// For simplicity, this test reads an exact copy of
// libiberty/testsuite/demangle-expected from GCC.  See that file for
// the syntax.  We ignore all tests that are not --format=gnu-v3 or
// --format=auto with a string starting with _Z.
func TestExpected(t *testing.T) {
	f, err := os.Open(filename)
	if err != nil {
		t.Fatal(err)
	}
	scanner := bufio.NewScanner(f)
	lineno := 1
	for {
		format, got := getOptLine(t, scanner, &lineno)
		if !got {
			break
		}
		report := lineno
		input := getLine(t, scanner, &lineno)
		expect := getLine(t, scanner, &lineno)

		testNoParams := false
		skip := false
		if len(format) > 0 && format[0] == '-' {
			for _, arg := range strings.Fields(format) {
				switch arg {
				case "--format=gnu-v3":
				case "--format=auto":
					if !strings.HasPrefix(input, "_Z") {
						skip = true
					}
				case "--no-params":
					testNoParams = true
				case "--ret-postfix", "--ret-drop":
					skip = true
				case "--is-v3-ctor", "--is-v3-dtor":
					skip = true
				default:
					if !strings.HasPrefix(arg, "--format=") {
						t.Errorf("%s:%d: unrecognized argument %s", filename, report, arg)
					}
					skip = true
				}
			}
		}

		// The libiberty testsuite passes DMGL_TYPES to
		// demangle type names, but that doesn't seem useful
		// and we don't support it.
		if !strings.HasPrefix(input, "_Z") && !strings.HasPrefix(input, "_GLOBAL_") {
			skip = true
		}

		var expectNoParams string
		if testNoParams {
			expectNoParams = getLine(t, scanner, &lineno)
		}

		if skip {
			continue
		}

		oneTest(t, report, input, expect, true)
		if testNoParams {
			oneTest(t, report, input, expectNoParams, false)
		}
	}
	if err := scanner.Err(); err != nil {
		t.Error(err)
	}
}

// oneTest tests one entry from demangle-expected.
func oneTest(t *testing.T, report int, input, expect string, params bool) {
	if *verbose {
		fmt.Println(input)
	}

	exception := exceptions[input]

	var s string
	var err error
	if params {
		s, err = ToString(input)
	} else {
		s, err = ToString(input, NoParams)
	}
	if err != nil {
		if exception {
			t.Logf("%s:%d: ignore expected difference: got %q, expected %q", filename, report, err, expect)
			return
		}

		if err != ErrNotMangledName {
			if input == expect {
				return
			}
			t.Errorf("%s:%d: %v", filename, report, err)
			return
		}
		s = input
	}

	if s != expect {
		if exception {
			t.Logf("%s:%d: ignore expected difference: got %q, expected %q", filename, report, s, expect)
		} else {
			var a AST
			if params {
				a, err = ToAST(input)
			} else {
				a, err = ToAST(input, NoParams)
			}
			if err != nil {
				t.Logf("ToAST error: %v", err)
			} else {
				t.Logf("\n%#v", a)
			}
			t.Errorf("%s:%d: params: %t: got %q, expected %q", filename, report, params, s, expect)
		}
	} else if exception && params {
		t.Errorf("%s:%d: unexpected success (input listed in exceptions)", filename, report)
	}
}

// getLine reads a line from demangle-expected.
func getLine(t *testing.T, scanner *bufio.Scanner, lineno *int) string {
	s, got := getOptLine(t, scanner, lineno)
	if !got {
		t.Fatalf("%s:%d: unexpected EOF", filename, *lineno)
	}
	return s
}

// getOptLine reads an optional line from demangle-expected, returning
// false at EOF.  It skips comment lines and updates *lineno.
func getOptLine(t *testing.T, scanner *bufio.Scanner, lineno *int) (string, bool) {
	for {
		if !scanner.Scan() {
			return "", false
		}
		*lineno++
		line := scanner.Text()
		if !strings.HasPrefix(line, "#") {
			return line, true
		}
	}
}
