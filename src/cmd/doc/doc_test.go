// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"flag"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

func TestMain(m *testing.M) {
	// Clear GOPATH so we don't access the user's own packages in the test.
	buildCtx.GOPATH = ""
	testGOPATH = true // force GOPATH mode; module test is in cmd/go/testdata/script/mod_doc.txt

	// Add $GOROOT/src/cmd/doc/testdata explicitly so we can access its contents in the test.
	// Normally testdata directories are ignored, but sending it to dirs.scan directly is
	// a hack that works around the check.
	testdataDir, err := filepath.Abs("testdata")
	if err != nil {
		panic(err)
	}
	dirsInit(
		Dir{importPath: "testdata", dir: testdataDir},
		Dir{importPath: "testdata/nested", dir: filepath.Join(testdataDir, "nested")},
		Dir{importPath: "testdata/nested/nested", dir: filepath.Join(testdataDir, "nested", "nested")})

	os.Exit(m.Run())
}

func maybeSkip(t *testing.T) {
	if runtime.GOOS == "ios" {
		t.Skip("iOS does not have a full file tree")
	}
}

type isDotSlashTest struct {
	str    string
	result bool
}

var isDotSlashTests = []isDotSlashTest{
	{``, false},
	{`x`, false},
	{`...`, false},
	{`.../`, false},
	{`...\`, false},

	{`.`, true},
	{`./`, true},
	{`.\`, true},
	{`./x`, true},
	{`.\x`, true},

	{`..`, true},
	{`../`, true},
	{`..\`, true},
	{`../x`, true},
	{`..\x`, true},
}

func TestIsDotSlashPath(t *testing.T) {
	for _, test := range isDotSlashTests {
		if result := isDotSlash(test.str); result != test.result {
			t.Errorf("isDotSlash(%q) = %t; expected %t", test.str, result, test.result)
		}
	}
}

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
			`const ExportedConstant = 1`,                                   // Simple constant.
			`const ConstOne = 1`,                                           // First entry in constant block.
			`const ConstFive ...`,                                          // From block starting with unexported constant.
			`var ExportedVariable = 1`,                                     // Simple variable.
			`var VarOne = 1`,                                               // First entry in variable block.
			`func ExportedFunc\(a int\) bool`,                              // Function.
			`func ReturnUnexported\(\) unexportedType`,                     // Function with unexported return type.
			`type ExportedType struct{ ... }`,                              // Exported type.
			`const ExportedTypedConstant ExportedType = iota`,              // Typed constant.
			`const ExportedTypedConstant_unexported unexportedType`,        // Typed constant, exported for unexported type.
			`const ConstLeft2 uint64 ...`,                                  // Typed constant using unexported iota.
			`const ConstGroup1 unexportedType = iota ...`,                  // Typed constant using unexported type.
			`const ConstGroup4 ExportedType = ExportedType{}`,              // Typed constant using exported type.
			`const MultiLineConst = ...`,                                   // Multi line constant.
			`var MultiLineVar = map\[struct{ ... }\]struct{ ... }{ ... }`,  // Multi line variable.
			`func MultiLineFunc\(x interface{ ... }\) \(r struct{ ... }\)`, // Multi line function.
			`var LongLine = newLongLine\(("someArgument[1-4]", ){4}...\)`,  // Long list of arguments.
			`type T1 = T2`,                                                 // Type alias
			`type SimpleConstraint interface{ ... }`,
			`type TildeConstraint interface{ ... }`,
			`type StructConstraint interface{ ... }`,
		},
		[]string{
			`const internalConstant = 2`,       // No internal constants.
			`var internalVariable = 2`,         // No internal variables.
			`func internalFunc(a int) bool`,    // No internal functions.
			`Comment about exported constant`,  // No comment for single constant.
			`Comment about exported variable`,  // No comment for single variable.
			`Comment about block of constants`, // No comment for constant block.
			`Comment about block of variables`, // No comment for variable block.
			`Comment before ConstOne`,          // No comment for first entry in constant block.
			`Comment before VarOne`,            // No comment for first entry in variable block.
			`ConstTwo = 2`,                     // No second entry in constant block.
			`VarTwo = 2`,                       // No second entry in variable block.
			`VarFive = 5`,                      // From block starting with unexported variable.
			`type unexportedType`,              // No unexported type.
			`unexportedTypedConstant`,          // No unexported typed constant.
			`\bField`,                          // No fields.
			`Method`,                           // No methods.
			`someArgument[5-8]`,                // No truncated arguments.
			`type T1 T2`,                       // Type alias does not display as type declaration.
		},
	},
	// Package dump -all
	{
		"full package",
		[]string{"-all", p},
		[]string{
			`package pkg .*import`,
			`Package comment`,
			`CONSTANTS`,
			`Comment before ConstOne`,
			`ConstOne = 1`,
			`ConstTwo = 2 // Comment on line with ConstTwo`,
			`ConstFive`,
			`ConstSix`,
			`Const block where first entry is unexported`,
			`ConstLeft2, constRight2 uint64`,
			`constLeft3, ConstRight3`,
			`ConstLeft4, ConstRight4`,
			`Duplicate = iota`,
			`const CaseMatch = 1`,
			`const Casematch = 2`,
			`const ExportedConstant = 1`,
			`const MultiLineConst = `,
			`MultiLineString1`,
			`VARIABLES`,
			`Comment before VarOne`,
			`VarOne = 1`,
			`Comment about block of variables`,
			`VarFive = 5`,
			`var ExportedVariable = 1`,
			`var ExportedVarOfUnExported unexportedType`,
			`var LongLine = newLongLine\(`,
			`var MultiLineVar = map\[struct {`,
			`FUNCTIONS`,
			`func ExportedFunc\(a int\) bool`,
			`Comment about exported function`,
			`func MultiLineFunc\(x interface`,
			`func ReturnUnexported\(\) unexportedType`,
			`TYPES`,
			`type ExportedInterface interface`,
			`type ExportedStructOneField struct`,
			`type ExportedType struct`,
			`Comment about exported type`,
			`const ConstGroup4 ExportedType = ExportedType`,
			`ExportedTypedConstant ExportedType = iota`,
			`Constants tied to ExportedType`,
			`func ExportedTypeConstructor\(\) \*ExportedType`,
			`Comment about constructor for exported type`,
			`func ReturnExported\(\) ExportedType`,
			`func \(ExportedType\) ExportedMethod\(a int\) bool`,
			`Comment about exported method`,
			`type T1 = T2`,
			`type T2 int`,
			`type SimpleConstraint interface {`,
			`type TildeConstraint interface {`,
			`type StructConstraint interface {`,
		},
		[]string{
			`constThree`,
			`_, _ uint64 = 2 \* iota, 1 << iota`,
			`constLeft1, constRight1`,
			`duplicate`,
			`varFour`,
			`func internalFunc`,
			`unexportedField`,
			`func \(unexportedType\)`,
		},
	},
	// Package with just the package declaration. Issue 31457.
	{
		"only package declaration",
		[]string{"-all", p + "/nested/empty"},
		[]string{`package empty .*import`},
		nil,
	},
	// Package dump -short
	{
		"full package with -short",
		[]string{`-short`, p},
		[]string{
			`const ExportedConstant = 1`,               // Simple constant.
			`func ReturnUnexported\(\) unexportedType`, // Function with unexported return type.
		},
		[]string{
			`MultiLine(String|Method|Field)`, // No data from multi line portions.
		},
	},
	// Package dump -u
	{
		"full package with u",
		[]string{`-u`, p},
		[]string{
			`const ExportedConstant = 1`,               // Simple constant.
			`const internalConstant = 2`,               // Internal constants.
			`func internalFunc\(a int\) bool`,          // Internal functions.
			`func ReturnUnexported\(\) unexportedType`, // Function with unexported return type.
		},
		[]string{
			`Comment about exported constant`,  // No comment for simple constant.
			`Comment about block of constants`, // No comment for constant block.
			`Comment about internal function`,  // No comment for internal function.
			`MultiLine(String|Method|Field)`,   // No data from multi line portions.
		},
	},
	// Package dump -u -all
	{
		"full package",
		[]string{"-u", "-all", p},
		[]string{
			`package pkg .*import`,
			`Package comment`,
			`CONSTANTS`,
			`Comment before ConstOne`,
			`ConstOne += 1`,
			`ConstTwo += 2 // Comment on line with ConstTwo`,
			`constThree = 3 // Comment on line with constThree`,
			`ConstFive`,
			`const internalConstant += 2`,
			`Comment about internal constant`,
			`VARIABLES`,
			`Comment before VarOne`,
			`VarOne += 1`,
			`Comment about block of variables`,
			`varFour += 4`,
			`VarFive += 5`,
			`varSix += 6`,
			`var ExportedVariable = 1`,
			`var LongLine = newLongLine\(`,
			`var MultiLineVar = map\[struct {`,
			`var internalVariable = 2`,
			`Comment about internal variable`,
			`FUNCTIONS`,
			`func ExportedFunc\(a int\) bool`,
			`Comment about exported function`,
			`func MultiLineFunc\(x interface`,
			`func internalFunc\(a int\) bool`,
			`Comment about internal function`,
			`func newLongLine\(ss .*string\)`,
			`TYPES`,
			`type ExportedType struct`,
			`type T1 = T2`,
			`type T2 int`,
			`type unexportedType int`,
			`Comment about unexported type`,
			`ConstGroup1 unexportedType = iota`,
			`ConstGroup2`,
			`ConstGroup3`,
			`ExportedTypedConstant_unexported unexportedType = iota`,
			`Constants tied to unexportedType`,
			`const unexportedTypedConstant unexportedType = 1`,
			`func ReturnUnexported\(\) unexportedType`,
			`func \(unexportedType\) ExportedMethod\(\) bool`,
			`func \(unexportedType\) unexportedMethod\(\) bool`,
		},
		nil,
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
	// Block of constants -src.
	{
		"block of constants with -src",
		[]string{"-src", p, `ConstTwo`},
		[]string{
			`Comment about block of constants`, // Top comment.
			`ConstOne.*=.*1`,                   // Each constant seen.
			`ConstTwo.*=.*2.*Comment on line with ConstTwo`,
			`constThree`, // Even unexported constants.
		},
		nil,
	},
	// Block of constants with carryover type from unexported field.
	{
		"block of constants with carryover type",
		[]string{p, `ConstLeft2`},
		[]string{
			`ConstLeft2, constRight2 uint64`,
			`constLeft3, ConstRight3`,
			`ConstLeft4, ConstRight4`,
		},
		nil,
	},
	// Block of constants -u with carryover type from unexported field.
	{
		"block of constants with carryover type",
		[]string{"-u", p, `ConstLeft2`},
		[]string{
			`_, _ uint64 = 2 \* iota, 1 << iota`,
			`constLeft1, constRight1`,
			`ConstLeft2, constRight2`,
			`constLeft3, ConstRight3`,
			`ConstLeft4, ConstRight4`,
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
	// Function with -src.
	{
		"function with -src",
		[]string{"-src", p, `ExportedFunc`},
		[]string{
			`Comment about exported function`, // Include comment.
			`func ExportedFunc\(a int\) bool`,
			`return true != false`, // Include body.
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
			`Comment before exported field.*\n.*ExportedField +int` +
				`.*Comment on line with exported field`,
			`ExportedEmbeddedType.*Comment on line with exported embedded field`,
			`Has unexported fields`,
			`func \(ExportedType\) ExportedMethod\(a int\) bool`,
			`const ExportedTypedConstant ExportedType = iota`, // Must include associated constant.
			`func ExportedTypeConstructor\(\) \*ExportedType`, // Must include constructor.
			`io.Reader.*Comment on line with embedded Reader`,
		},
		[]string{
			`unexportedField`,               // No unexported field.
			`int.*embedded`,                 // No unexported embedded field.
			`Comment about exported method`, // No comment about exported method.
			`unexportedMethod`,              // No unexported method.
			`unexportedTypedConstant`,       // No unexported constant.
			`error`,                         // No embedded error.
		},
	},
	// Type with -src. Will see unexported fields.
	{
		"type",
		[]string{"-src", p, `ExportedType`},
		[]string{
			`Comment about exported type`, // Include comment.
			`type ExportedType struct`,    // Type definition.
			`Comment before exported field`,
			`ExportedField.*Comment on line with exported field`,
			`ExportedEmbeddedType.*Comment on line with exported embedded field`,
			`unexportedType.*Comment on line with unexported embedded field`,
			`func \(ExportedType\) ExportedMethod\(a int\) bool`,
			`const ExportedTypedConstant ExportedType = iota`, // Must include associated constant.
			`func ExportedTypeConstructor\(\) \*ExportedType`, // Must include constructor.
			`io.Reader.*Comment on line with embedded Reader`,
		},
		[]string{
			`Comment about exported method`, // No comment about exported method.
			`unexportedMethod`,              // No unexported method.
			`unexportedTypedConstant`,       // No unexported constant.
		},
	},
	// Type -all.
	{
		"type",
		[]string{"-all", p, `ExportedType`},
		[]string{
			`type ExportedType struct {`,                        // Type definition as source.
			`Comment about exported type`,                       // Include comment afterwards.
			`const ConstGroup4 ExportedType = ExportedType\{\}`, // Related constants.
			`ExportedTypedConstant ExportedType = iota`,
			`Constants tied to ExportedType`,
			`func ExportedTypeConstructor\(\) \*ExportedType`,
			`Comment about constructor for exported type.`,
			`func ReturnExported\(\) ExportedType`,
			`func \(ExportedType\) ExportedMethod\(a int\) bool`,
			`Comment about exported method.`,
			`func \(ExportedType\) Uncommented\(a int\) bool\n\n`, // Ensure line gap after method with no comment
		},
		[]string{
			`unexportedType`,
		},
	},
	// Type T1 dump (alias).
	{
		"type T1",
		[]string{p + ".T1"},
		[]string{
			`type T1 = T2`,
		},
		[]string{
			`type T1 T2`,
			`type ExportedType`,
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
			`unexportedField.*int.*Comment on line with unexported field`,
			`ExportedEmbeddedType.*Comment on line with exported embedded field`,
			`\*ExportedEmbeddedType.*Comment on line with exported embedded \*field`,
			`\*qualified.ExportedEmbeddedType.*Comment on line with exported embedded \*selector.field`,
			`unexportedType.*Comment on line with unexported embedded field`,
			`\*unexportedType.*Comment on line with unexported embedded \*field`,
			`io.Reader.*Comment on line with embedded Reader`,
			`error.*Comment on line with embedded error`,
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

	// Interface.
	{
		"interface type",
		[]string{p, `ExportedInterface`},
		[]string{
			`Comment about exported interface`, // Include comment.
			`type ExportedInterface interface`, // Interface definition.
			`Comment before exported method.\n.*//\n.*//	// Code block showing how to use ExportedMethod\n.*//	func DoSomething\(\) error {\n.*//		ExportedMethod\(\)\n.*//		return nil\n.*//	}\n.*//.*\n.*ExportedMethod\(\)` +
				`.*Comment on line with exported method`,
			`io.Reader.*Comment on line with embedded Reader`,
			`error.*Comment on line with embedded error`,
			`Has unexported methods`,
		},
		[]string{
			`unexportedField`,               // No unexported field.
			`Comment about exported method`, // No comment about exported method.
			`unexportedMethod`,              // No unexported method.
			`unexportedTypedConstant`,       // No unexported constant.
		},
	},
	// Interface -u with unexported methods.
	{
		"interface type with unexported methods and -u",
		[]string{"-u", p, `ExportedInterface`},
		[]string{
			`Comment about exported interface`, // Include comment.
			`type ExportedInterface interface`, // Interface definition.
			`Comment before exported method.\n.*//\n.*//	// Code block showing how to use ExportedMethod\n.*//	func DoSomething\(\) error {\n.*//		ExportedMethod\(\)\n.*//		return nil\n.*//	}\n.*//.*\n.*ExportedMethod\(\)` + `.*Comment on line with exported method`,
			`unexportedMethod\(\).*Comment on line with unexported method`,
			`io.Reader.*Comment on line with embedded Reader`,
			`error.*Comment on line with embedded error`,
		},
		[]string{
			`Has unexported methods`,
		},
	},

	// Interface method.
	{
		"interface method",
		[]string{p, `ExportedInterface.ExportedMethod`},
		[]string{
			`Comment before exported method.\n.*//\n.*//	// Code block showing how to use ExportedMethod\n.*//	func DoSomething\(\) error {\n.*//		ExportedMethod\(\)\n.*//		return nil\n.*//	}\n.*//.*\n.*ExportedMethod\(\)` +
				`.*Comment on line with exported method`,
		},
		[]string{
			`Comment about exported interface`,
		},
	},
	// Interface method at package level.
	{
		"interface method at package level",
		[]string{p, `ExportedMethod`},
		[]string{
			`func \(ExportedType\) ExportedMethod\(a int\) bool`,
			`Comment about exported method`,
		},
		[]string{
			`Comment before exported method.*\n.*ExportedMethod\(\)` +
				`.*Comment on line with exported method`,
		},
	},

	// Method.
	{
		"method",
		[]string{p, `ExportedType.ExportedMethod`},
		[]string{
			`func \(ExportedType\) ExportedMethod\(a int\) bool`,
			`Comment about exported method`,
		},
		nil,
	},
	// Method  with -u.
	{
		"method with -u",
		[]string{"-u", p, `ExportedType.unexportedMethod`},
		[]string{
			`func \(ExportedType\) unexportedMethod\(a int\) bool`,
			`Comment about unexported method`,
		},
		nil,
	},
	// Method with -src.
	{
		"method with -src",
		[]string{"-src", p, `ExportedType.ExportedMethod`},
		[]string{
			`func \(ExportedType\) ExportedMethod\(a int\) bool`,
			`Comment about exported method`,
			`return true != true`,
		},
		nil,
	},

	// Field.
	{
		"field",
		[]string{p, `ExportedType.ExportedField`},
		[]string{
			`type ExportedType struct`,
			`ExportedField int`,
			`Comment before exported field`,
			`Comment on line with exported field`,
			`other fields elided`,
		},
		nil,
	},

	// Field with -u.
	{
		"method with -u",
		[]string{"-u", p, `ExportedType.unexportedField`},
		[]string{
			`unexportedField int`,
			`Comment on line with unexported field`,
		},
		nil,
	},

	// Field of struct with only one field.
	{
		"single-field struct",
		[]string{p, `ExportedStructOneField.OnlyField`},
		[]string{`the only field`},
		[]string{`other fields elided`},
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

	// Merging comments with -src.
	{
		"merge comments with -src A",
		[]string{"-src", p + "/merge", `A`},
		[]string{
			`A doc`,
			`func A`,
			`A comment`,
		},
		[]string{
			`Package A doc`,
			`Package B doc`,
			`B doc`,
			`B comment`,
			`B doc`,
		},
	},
	{
		"merge comments with -src B",
		[]string{"-src", p + "/merge", `B`},
		[]string{
			`B doc`,
			`func B`,
			`B comment`,
		},
		[]string{
			`Package A doc`,
			`Package B doc`,
			`A doc`,
			`A comment`,
			`A doc`,
		},
	},

	// No dups with -u. Issue 21797.
	{
		"case matching on, no dups",
		[]string{"-u", p, `duplicate`},
		[]string{
			`Duplicate`,
			`duplicate`,
		},
		[]string{
			"\\)\n+const", // This will appear if the const decl appears twice.
		},
	},
	{
		"non-imported: pkg.sym",
		[]string{"nested.Foo"},
		[]string{"Foo struct"},
		nil,
	},
	{
		"non-imported: pkg only",
		[]string{"nested"},
		[]string{"Foo struct"},
		nil,
	},
	{
		"non-imported: pkg sym",
		[]string{"nested", "Foo"},
		[]string{"Foo struct"},
		nil,
	},
	{
		"formatted doc on function",
		[]string{p, "ExportedFormattedDoc"},
		[]string{
			`func ExportedFormattedDoc\(a int\) bool`,
			`    Comment about exported function with formatting\.

    Example

        fmt\.Println\(FormattedDoc\(\)\)

    Text after pre-formatted block\.`,
		},
		nil,
	},
	{
		"formatted doc on type field",
		[]string{p, "ExportedFormattedType.ExportedField"},
		[]string{
			`type ExportedFormattedType struct`,
			`    // Comment before exported field with formatting\.
    //[ ]
    // Example
    //[ ]
    //     a\.ExportedField = 123
    //[ ]
    // Text after pre-formatted block\.`,
			`ExportedField int`,
		},
		nil,
	},
}

func TestDoc(t *testing.T) {
	maybeSkip(t)
	defer log.SetOutput(log.Writer())
	for _, test := range tests {
		var b bytes.Buffer
		var flagSet flag.FlagSet
		var logbuf bytes.Buffer
		log.SetOutput(&logbuf)
		err := do(&b, &flagSet, test.args)
		if err != nil {
			t.Fatalf("%s %v: %s\n", test.name, test.args, err)
		}
		if logbuf.Len() > 0 {
			t.Errorf("%s %v: unexpected log messages:\n%s", test.name, test.args, logbuf.Bytes())
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
		if bytes.Count(output, []byte("TYPES\n")) > 1 {
			t.Fatalf("%s: repeating headers", test.name)
		}
		if failed {
			t.Logf("\n%s", output)
		}
	}
}

// Test the code to try multiple packages. Our test case is
//	go doc rand.Float64
// This needs to find math/rand.Float64; however crypto/rand, which doesn't
// have the symbol, usually appears first in the directory listing.
func TestMultiplePackages(t *testing.T) {
	if testing.Short() {
		t.Skip("scanning file system takes too long")
	}
	maybeSkip(t)
	var b bytes.Buffer // We don't care about the output.
	// Make sure crypto/rand does not have the symbol.
	{
		var flagSet flag.FlagSet
		err := do(&b, &flagSet, []string{"crypto/rand.float64"})
		if err == nil {
			t.Errorf("expected error from crypto/rand.float64")
		} else if !strings.Contains(err.Error(), "no symbol float64") {
			t.Errorf("unexpected error %q from crypto/rand.float64", err)
		}
	}
	// Make sure math/rand does have the symbol.
	{
		var flagSet flag.FlagSet
		err := do(&b, &flagSet, []string{"math/rand.float64"})
		if err != nil {
			t.Errorf("unexpected error %q from math/rand.float64", err)
		}
	}
	// Try the shorthand.
	{
		var flagSet flag.FlagSet
		err := do(&b, &flagSet, []string{"rand.float64"})
		if err != nil {
			t.Errorf("unexpected error %q from rand.float64", err)
		}
	}
	// Now try a missing symbol. We should see both packages in the error.
	{
		var flagSet flag.FlagSet
		err := do(&b, &flagSet, []string{"rand.doesnotexit"})
		if err == nil {
			t.Errorf("expected error from rand.doesnotexit")
		} else {
			errStr := err.Error()
			if !strings.Contains(errStr, "no symbol") {
				t.Errorf("error %q should contain 'no symbol", errStr)
			}
			if !strings.Contains(errStr, "crypto/rand") {
				t.Errorf("error %q should contain crypto/rand", errStr)
			}
			if !strings.Contains(errStr, "math/rand") {
				t.Errorf("error %q should contain math/rand", errStr)
			}
		}
	}
}

// Test the code to look up packages when given two args. First test case is
//	go doc binary BigEndian
// This needs to find encoding/binary.BigEndian, which means
// finding the package encoding/binary given only "binary".
// Second case is
//	go doc rand Float64
// which again needs to find math/rand and not give up after crypto/rand,
// which has no such function.
func TestTwoArgLookup(t *testing.T) {
	if testing.Short() {
		t.Skip("scanning file system takes too long")
	}
	maybeSkip(t)
	var b bytes.Buffer // We don't care about the output.
	{
		var flagSet flag.FlagSet
		err := do(&b, &flagSet, []string{"binary", "BigEndian"})
		if err != nil {
			t.Errorf("unexpected error %q from binary BigEndian", err)
		}
	}
	{
		var flagSet flag.FlagSet
		err := do(&b, &flagSet, []string{"rand", "Float64"})
		if err != nil {
			t.Errorf("unexpected error %q from rand Float64", err)
		}
	}
	{
		var flagSet flag.FlagSet
		err := do(&b, &flagSet, []string{"bytes", "Foo"})
		if err == nil {
			t.Errorf("expected error from bytes Foo")
		} else if !strings.Contains(err.Error(), "no symbol Foo") {
			t.Errorf("unexpected error %q from bytes Foo", err)
		}
	}
	{
		var flagSet flag.FlagSet
		err := do(&b, &flagSet, []string{"nosuchpackage", "Foo"})
		if err == nil {
			// actually present in the user's filesystem
		} else if !strings.Contains(err.Error(), "no such package") {
			t.Errorf("unexpected error %q from nosuchpackage Foo", err)
		}
	}
}

// Test the code to look up packages when the first argument starts with "./".
// Our test case is in effect "cd src/text; doc ./template". This should get
// text/template but before Issue 23383 was fixed would give html/template.
func TestDotSlashLookup(t *testing.T) {
	if testing.Short() {
		t.Skip("scanning file system takes too long")
	}
	maybeSkip(t)
	where, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.Chdir(where); err != nil {
			t.Fatal(err)
		}
	}()
	if err := os.Chdir(filepath.Join(buildCtx.GOROOT, "src", "text")); err != nil {
		t.Fatal(err)
	}
	var b bytes.Buffer
	var flagSet flag.FlagSet
	err = do(&b, &flagSet, []string{"./template"})
	if err != nil {
		t.Errorf("unexpected error %q from ./template", err)
	}
	// The output should contain information about the text/template package.
	const want = `package template // import "text/template"`
	output := b.String()
	if !strings.HasPrefix(output, want) {
		t.Fatalf("wrong package: %.*q...", len(want), output)
	}
}

// Test that we don't print spurious package clauses
// when there should be no output at all. Issue 37969.
func TestNoPackageClauseWhenNoMatch(t *testing.T) {
	maybeSkip(t)
	var b bytes.Buffer
	var flagSet flag.FlagSet
	err := do(&b, &flagSet, []string{"template.ZZZ"})
	// Expect an error.
	if err == nil {
		t.Error("expect an error for template.zzz")
	}
	// And the output should not contain any package clauses.
	const dontWant = `package template // import `
	output := b.String()
	if strings.Contains(output, dontWant) {
		t.Fatalf("improper package clause printed:\n%s", output)
	}
}

type trimTest struct {
	path   string
	prefix string
	result string
	ok     bool
}

var trimTests = []trimTest{
	{"", "", "", true},
	{"/usr/gopher", "/usr/gopher", "/usr/gopher", true},
	{"/usr/gopher/bar", "/usr/gopher", "bar", true},
	{"/usr/gopherflakes", "/usr/gopher", "/usr/gopherflakes", false},
	{"/usr/gopher/bar", "/usr/zot", "/usr/gopher/bar", false},
}

func TestTrim(t *testing.T) {
	for _, test := range trimTests {
		result, ok := trim(test.path, test.prefix)
		if ok != test.ok {
			t.Errorf("%s %s expected %t got %t", test.path, test.prefix, test.ok, ok)
			continue
		}
		if result != test.result {
			t.Errorf("%s %s expected %q got %q", test.path, test.prefix, test.result, result)
			continue
		}
	}
}
