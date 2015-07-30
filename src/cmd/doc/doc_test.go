// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"flag"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"testing"
)

const (
	dataDir = "testdata"
	binary  = "testdoc"
)

type test struct {
	name string
	args []string // Arguments to "[go] doc".
	yes  []string // Regular expressions that should match.
	no   []string // Regular expressions that should not match.
}

const p = "cmd/doc/testdata"

var tests = []test{
	// Sanity check.
	{
		"sanity check",
		[]string{p},
		[]string{`type ExportedType struct`},
		nil,
	},

	// Package dump includes import, package statement.
	{
		"package clause",
		[]string{p},
		[]string{`package pkg.*cmd/doc/testdata`},
		nil,
	},

	// Constants.
	// Package dump
	{
		"full package",
		[]string{p},
		[]string{
			`Package comment`,
			`const ExportedConstant = 1`,                            // Simple constant.
			`const ConstOne = 1`,                                    // First entry in constant block.
			`var ExportedVariable = 1`,                              // Simple variable.
			`var VarOne = 1`,                                        // First entry in variable block.
			`func ExportedFunc\(a int\) bool`,                       // Function.
			`type ExportedType struct { ... }`,                      // Exported type.
			`const ExportedTypedConstant ExportedType = iota`,       // Typed constant.
			`const ExportedTypedConstant_unexported unexportedType`, // Typed constant, exported for unexported type.
		},
		[]string{
			`const internalConstant = 2`,        // No internal constants.
			`var internalVariable = 2`,          // No internal variables.
			`func internalFunc(a int) bool`,     // No internal functions.
			`Comment about exported constant`,   // No comment for single constant.
			`Comment about exported variable`,   // No comment for single variable.
			`Comment about block of constants.`, // No comment for constant block.
			`Comment about block of variables.`, // No comment for variable block.
			`Comment before ConstOne`,           // No comment for first entry in constant block.
			`Comment before VarOne`,             // No comment for first entry in variable block.
			`ConstTwo = 2`,                      // No second entry in constant block.
			`VarTwo = 2`,                        // No second entry in variable block.
			`type unexportedType`,               // No unexported type.
			`unexportedTypedConstant`,           // No unexported typed constant.
			`Field`,                             // No fields.
			`Method`,                            // No methods.
		},
	},
	// Package dump -u
	{
		"full package with u",
		[]string{`-u`, p},
		[]string{
			`const ExportedConstant = 1`,      // Simple constant.
			`const internalConstant = 2`,      // Internal constants.
			`func internalFunc\(a int\) bool`, // Internal functions.
		},
		[]string{
			`Comment about exported constant`,  // No comment for simple constant.
			`Comment about block of constants`, // No comment for constant block.
			`Comment about internal function`,  // No comment for internal function.
		},
	},

	// Single constant.
	{
		"single constant",
		[]string{p, `ExportedConstant`},
		[]string{
			`Comment about exported constant`, // Include comment.
			`const ExportedConstant = 1`,
		},
		nil,
	},
	// Single constant -u.
	{
		"single constant with -u",
		[]string{`-u`, p, `internalConstant`},
		[]string{
			`Comment about internal constant`, // Include comment.
			`const internalConstant = 2`,
		},
		nil,
	},
	// Block of constants.
	{
		"block of constants",
		[]string{p, `ConstTwo`},
		[]string{
			`Comment before ConstOne.\n.*ConstOne = 1`,    // First...
			`ConstTwo = 2.*Comment on line with ConstTwo`, // And second show up.
			`Comment about block of constants`,            // Comment does too.
		},
		[]string{
			`constThree`, // No unexported constant.
		},
	},
	// Block of constants -u.
	{
		"block of constants with -u",
		[]string{"-u", p, `constThree`},
		[]string{
			`constThree = 3.*Comment on line with constThree`,
		},
		nil,
	},

	// Single variable.
	{
		"single variable",
		[]string{p, `ExportedVariable`},
		[]string{
			`ExportedVariable`, // Include comment.
			`var ExportedVariable = 1`,
		},
		nil,
	},
	// Single variable -u.
	{
		"single variable with -u",
		[]string{`-u`, p, `internalVariable`},
		[]string{
			`Comment about internal variable`, // Include comment.
			`var internalVariable = 2`,
		},
		nil,
	},
	// Block of variables.
	{
		"block of variables",
		[]string{p, `VarTwo`},
		[]string{
			`Comment before VarOne.\n.*VarOne = 1`,    // First...
			`VarTwo = 2.*Comment on line with VarTwo`, // And second show up.
			`Comment about block of variables`,        // Comment does too.
		},
		[]string{
			`varThree= 3`, // No unexported variable.
		},
	},
	// Block of variables -u.
	{
		"block of variables with -u",
		[]string{"-u", p, `varThree`},
		[]string{
			`varThree = 3.*Comment on line with varThree`,
		},
		nil,
	},

	// Function.
	{
		"function",
		[]string{p, `ExportedFunc`},
		[]string{
			`Comment about exported function`, // Include comment.
			`func ExportedFunc\(a int\) bool`,
		},
		nil,
	},
	// Function -u.
	{
		"function with -u",
		[]string{"-u", p, `internalFunc`},
		[]string{
			`Comment about internal function`, // Include comment.
			`func internalFunc\(a int\) bool`,
		},
		nil,
	},

	// Type.
	{
		"type",
		[]string{p, `ExportedType`},
		[]string{
			`Comment about exported type`, // Include comment.
			`type ExportedType struct`,    // Type definition.
			`Comment before exported field.*\n.*ExportedField +int`,
			`Has unexported fields`,
			`func \(ExportedType\) ExportedMethod\(a int\) bool`,
			`const ExportedTypedConstant ExportedType = iota`, // Must include associated constant.
			`func ExportedTypeConstructor\(\) \*ExportedType`, // Must include constructor.
		},
		[]string{
			`unexportedField`,                // No unexported field.
			`Comment about exported method.`, // No comment about exported method.
			`unexportedMethod`,               // No unexported method.
			`unexportedTypedConstant`,        // No unexported constant.
		},
	},
	// Type -u with unexported fields.
	{
		"type with unexported fields and -u",
		[]string{"-u", p, `ExportedType`},
		[]string{
			`Comment about exported type`, // Include comment.
			`type ExportedType struct`,    // Type definition.
			`Comment before exported field.*\n.*ExportedField +int`,
			`unexportedField int.*Comment on line with unexported field.`,
			`func \(ExportedType\) unexportedMethod\(a int\) bool`,
			`unexportedTypedConstant`,
		},
		[]string{
			`Has unexported fields`,
		},
	},
	// Unexported type with -u.
	{
		"unexported type with -u",
		[]string{"-u", p, `unexportedType`},
		[]string{
			`Comment about unexported type`, // Include comment.
			`type unexportedType int`,       // Type definition.
			`func \(unexportedType\) ExportedMethod\(\) bool`,
			`func \(unexportedType\) unexportedMethod\(\) bool`,
			`ExportedTypedConstant_unexported unexportedType = iota`,
			`const unexportedTypedConstant unexportedType = 1`,
		},
		nil,
	},

	// Method.
	{
		"method",
		[]string{p, `ExportedType.ExportedMethod`},
		[]string{
			`func \(ExportedType\) ExportedMethod\(a int\) bool`,
			`Comment about exported method.`,
		},
		nil,
	},
	// Method  with -u.
	{
		"method with -u",
		[]string{"-u", p, `ExportedType.unexportedMethod`},
		[]string{
			`func \(ExportedType\) unexportedMethod\(a int\) bool`,
			`Comment about unexported method.`,
		},
		nil,
	},

	// Case matching off.
	{
		"case matching off",
		[]string{p, `casematch`},
		[]string{
			`CaseMatch`,
			`Casematch`,
		},
		nil,
	},

	// Case matching on.
	{
		"case matching on",
		[]string{"-c", p, `Casematch`},
		[]string{
			`Casematch`,
		},
		[]string{
			`CaseMatch`,
		},
	},
}

func TestDoc(t *testing.T) {
	if runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64") {
		t.Skip("TODO: on darwin/arm, test fails: no such package cmd/doc/testdata")
	}
	for _, test := range tests {
		var b bytes.Buffer
		var flagSet flag.FlagSet
		err := do(&b, &flagSet, test.args)
		if err != nil {
			t.Fatalf("%s: %s\n", test.name, err)
		}
		output := b.Bytes()
		failed := false
		for j, yes := range test.yes {
			re, err := regexp.Compile(yes)
			if err != nil {
				t.Fatalf("%s.%d: compiling %#q: %s", test.name, j, yes, err)
			}
			if !re.Match(output) {
				t.Errorf("%s.%d: no match for %s %#q", test.name, j, test.args, yes)
				failed = true
			}
		}
		for j, no := range test.no {
			re, err := regexp.Compile(no)
			if err != nil {
				t.Fatalf("%s.%d: compiling %#q: %s", test.name, j, no, err)
			}
			if re.Match(output) {
				t.Errorf("%s.%d: incorrect match for %s %#q", test.name, j, test.args, no)
				failed = true
			}
		}
		if failed {
			t.Logf("\n%s", output)
		}
	}
}

// run runs the command, but calls t.Fatal if there is an error.
func run(c *exec.Cmd, t *testing.T) []byte {
	output, err := c.CombinedOutput()
	if err != nil {
		os.Stdout.Write(output)
		t.Fatal(err)
	}
	return output
}
