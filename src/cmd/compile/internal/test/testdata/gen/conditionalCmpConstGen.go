// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This program generates tests to verify that conditional comparisons
// with constants are properly optimized by the compiler through constant folding.
// The generated test should be compiled with a known working version of Go.
// Run with `go run conditionalCmpConstGen.go` to generate a file called
// conditionalCmpConst_test.go in the grandparent directory.

package main

import (
	"bytes"
	"fmt"
	"go/format"
	"log"
	"os"
	"strings"
)

// IntegerConstraint defines a type constraint for all integer types
func writeIntegerConstraint(w *bytes.Buffer) {
	fmt.Fprintf(w, "type IntegerConstraint interface {\n")
	fmt.Fprintf(w, "\tint | uint | int8 | uint8 | int16 | ")
	fmt.Fprintf(w, "uint16 | int32 | uint32 | int64 | uint64\n")
	fmt.Fprintf(w, "}\n\n")
}

// TestCase describes a parameterized test case with comparison and logical operations
func writeTestCaseStruct(w *bytes.Buffer) {
	fmt.Fprintf(w, "type TestCase[T IntegerConstraint] struct {\n")
	fmt.Fprintf(w, "\tcmp1, cmp2 func(a, b T) bool\n")
	fmt.Fprintf(w, "\tcombine func(x, y bool) bool\n")
	fmt.Fprintf(w, "\ttargetFunc func(a, b, c, d T) bool\n")
	fmt.Fprintf(w, "\tcmp1Expr, cmp2Expr, logicalExpr string // String representations for debugging\n")
	fmt.Fprintf(w, "}\n\n")
}

// BoundaryValues contains base value and its variations for edge case testing
func writeBoundaryValuesStruct(w *bytes.Buffer) {
	fmt.Fprintf(w, "type BoundaryValues[T IntegerConstraint] struct {\n")
	fmt.Fprintf(w, "\tbase T\n")
	fmt.Fprintf(w, "\tvariants [3]T\n")
	fmt.Fprintf(w, "}\n\n")
}

// writeTypeDefinitions generates all necessary type declarations
func writeTypeDefinitions(w *bytes.Buffer) {
	writeIntegerConstraint(w)
	writeTestCaseStruct(w)
	writeBoundaryValuesStruct(w)
}

// comparisonOperators contains format strings for comparison operators
var comparisonOperators = []string{
	"%s == %s", "%s <= %s", "%s < %s",
	"%s != %s", "%s >= %s", "%s > %s",
}

// logicalOperators contains format strings for logical combination of boolean expressions
var logicalOperators = []string{
	"(%s) && (%s)", "(%s) && !(%s)", "!(%s) && (%s)", "!(%s) && !(%s)",
	"(%s) || (%s)", "(%s) || !(%s)", "!(%s) || (%s)", "!(%s) || !(%s)",
}

// writeComparator generates a comparator function based on the comparison operator
func writeComparator(w *bytes.Buffer, fieldName, operatorFormat string) {
	expression := fmt.Sprintf(operatorFormat, "a", "b")
	fmt.Fprintf(w, "\t\t\t%s: func(a, b T) bool { return %s },\n", fieldName, expression)
}

// writeLogicalCombiner generates a function to combine two boolean values
func writeLogicalCombiner(w *bytes.Buffer, logicalOperator string) {
	expression := fmt.Sprintf(logicalOperator, "x", "y")
	fmt.Fprintf(w, "\t\t\tcombine: func(x, y bool) bool { return %s },\n", expression)
}

// writeTargetFunction generates the target function with conditional expression
func writeTargetFunction(w *bytes.Buffer, cmp1, cmp2, logicalOp string) {
	leftExpr := fmt.Sprintf(cmp1, "a", "b")
	rightExpr := fmt.Sprintf(cmp2, "c", "d")
	condition := fmt.Sprintf(logicalOp, leftExpr, rightExpr)

	fmt.Fprintf(w, "\t\t\ttargetFunc: func(a, b, c, d T) bool {\n")
	fmt.Fprintf(w, "\t\t\t\tif %s {\n", condition)
	fmt.Fprintf(w, "\t\t\t\t\treturn true\n")
	fmt.Fprintf(w, "\t\t\t\t}\n")
	fmt.Fprintf(w, "\t\t\t\treturn false\n")
	fmt.Fprintf(w, "\t\t\t},\n")
}

// writeTestCase creates a single test case with given comparison and logical operators
func writeTestCase(w *bytes.Buffer, cmp1, cmp2, logicalOp string) {
	fmt.Fprintf(w, "\t\t{\n")
	writeComparator(w, "cmp1", cmp1)
	writeComparator(w, "cmp2", cmp2)
	writeLogicalCombiner(w, logicalOp)
	writeTargetFunction(w, cmp1, cmp2, logicalOp)

	// Store string representations for debugging
	cmp1Expr := fmt.Sprintf(cmp1, "a", "b")
	cmp2Expr := fmt.Sprintf(cmp2, "c", "d")
	logicalExpr := fmt.Sprintf(logicalOp, cmp1Expr, cmp2Expr)

	fmt.Fprintf(w, "\t\t\tcmp1Expr: %q,\n", cmp1Expr)
	fmt.Fprintf(w, "\t\t\tcmp2Expr: %q,\n", cmp2Expr)
	fmt.Fprintf(w, "\t\t\tlogicalExpr: %q,\n", logicalExpr)

	fmt.Fprintf(w, "\t\t},\n")
}

// generateTestCases creates a slice of all possible test cases
func generateTestCases(w *bytes.Buffer) {
	fmt.Fprintf(w, "func generateTestCases[T IntegerConstraint]() []TestCase[T] {\n")
	fmt.Fprintf(w, "\treturn []TestCase[T]{\n")

	for _, cmp1 := range comparisonOperators {
		for _, cmp2 := range comparisonOperators {
			for _, logicalOp := range logicalOperators {
				writeTestCase(w, cmp1, cmp2, logicalOp)
			}
		}
	}

	fmt.Fprintf(w, "\t}\n")
	fmt.Fprintf(w, "}\n\n")
}

// TypeConfig defines a type and its test base value
type TypeConfig struct {
	typeName, baseValue string
}

// typeConfigs contains all integer types to test with their base values
var typeConfigs = []TypeConfig{
	{typeName: "int8", baseValue: "1 << 6"},
	{typeName: "uint8", baseValue: "1 << 6"},
	{typeName: "int16", baseValue: "1 << 14"},
	{typeName: "uint16", baseValue: "1 << 14"},
	{typeName: "int32", baseValue: "1 << 30"},
	{typeName: "uint32", baseValue: "1 << 30"},
	{typeName: "int", baseValue: "1 << 30"},
	{typeName: "uint", baseValue: "1 << 30"},
	{typeName: "int64", baseValue: "1 << 62"},
	{typeName: "uint64", baseValue: "1 << 62"},
}

// writeTypeSpecificTest generates test for a specific integer type
func writeTypeSpecificTest(w *bytes.Buffer, typeName, baseValue string) {
	typeTitle := strings.Title(typeName)

	fmt.Fprintf(w, "func Test%sConditionalCmpConst(t *testing.T) {\n", typeTitle)

	fmt.Fprintf(w, "\ttestCases := generateTestCases[%s]()\n", typeName)
	fmt.Fprintf(w, "\tbase := %s(%s)\n", typeName, baseValue)
	fmt.Fprintf(w, "\tvalues := [3]%s{base - 1, base, base + 1}\n\n", typeName)

	fmt.Fprintf(w, "\tfor _, tc := range testCases {\n")
	fmt.Fprintf(w, "\t\ta, c := base, base\n")
	fmt.Fprintf(w, "\t\tfor _, b := range values {\n")
	fmt.Fprintf(w, "\t\t\tfor _, d := range values {\n")
	fmt.Fprintf(w, "\t\t\t\texpected := tc.combine(tc.cmp1(a, b), tc.cmp2(c, d))\n")
	fmt.Fprintf(w, "\t\t\t\tactual := tc.targetFunc(a, b, c, d)\n")
	fmt.Fprintf(w, "\t\t\t\tif actual != expected {\n")
	fmt.Fprintf(w, "\t\t\t\t\tt.Errorf(\"conditional comparison failed:\\n\"+\n")
	fmt.Fprintf(w, "\t\t\t\t\t\t\"  type: %%T\\n\"+\n")
	fmt.Fprintf(w, "\t\t\t\t\t\t\"  condition: %%s\\n\"+\n")
	fmt.Fprintf(w, "\t\t\t\t\t\t\"  values: a=%%v, b=%%v, c=%%v, d=%%v\\n\"+\n")
	fmt.Fprintf(w, "\t\t\t\t\t\t\"  cmp1(a,b)=%%v (%%s)\\n\"+\n")
	fmt.Fprintf(w, "\t\t\t\t\t\t\"  cmp2(c,d)=%%v (%%s)\\n\"+\n")
	fmt.Fprintf(w, "\t\t\t\t\t\t\"  expected: combine(%%v, %%v)=%%v\\n\"+\n")
	fmt.Fprintf(w, "\t\t\t\t\t\t\"  actual: %%v\\n\"+\n")
	fmt.Fprintf(w, "\t\t\t\t\t\t\"  logical expression: %%s\",\n")
	fmt.Fprintf(w, "\t\t\t\t\t\ta,\n")
	fmt.Fprintf(w, "\t\t\t\t\t\ttc.logicalExpr,\n")
	fmt.Fprintf(w, "\t\t\t\t\t\ta, b, c, d,\n")
	fmt.Fprintf(w, "\t\t\t\t\t\ttc.cmp1(a, b), tc.cmp1Expr,\n")
	fmt.Fprintf(w, "\t\t\t\t\t\ttc.cmp2(c, d), tc.cmp2Expr,\n")
	fmt.Fprintf(w, "\t\t\t\t\t\ttc.cmp1(a, b), tc.cmp2(c, d), expected,\n")
	fmt.Fprintf(w, "\t\t\t\t\t\tactual,\n")
	fmt.Fprintf(w, "\t\t\t\t\t\ttc.logicalExpr)\n")
	fmt.Fprintf(w, "\t\t\t\t}\n")
	fmt.Fprintf(w, "\t\t\t}\n")
	fmt.Fprintf(w, "\t\t}\n")
	fmt.Fprintf(w, "\t}\n")

	fmt.Fprintf(w, "}\n\n")
}

// writeAllTests generates tests for all supported integer types
func writeAllTests(w *bytes.Buffer) {
	for _, config := range typeConfigs {
		writeTypeSpecificTest(w, config.typeName, config.baseValue)
	}
}

func main() {
	buffer := new(bytes.Buffer)

	// Header for generated file
	fmt.Fprintf(buffer, "// Code generated by conditionalCmpConstGen.go; DO NOT EDIT.\n\n")
	fmt.Fprintf(buffer, "package test\n\n")
	fmt.Fprintf(buffer, "import \"testing\"\n\n")

	// Generate type definitions
	writeTypeDefinitions(buffer)

	// Generate test cases
	generateTestCases(buffer)

	// Generate specific tests for each integer type
	writeAllTests(buffer)

	// Format generated source code
	rawSource := buffer.Bytes()
	formattedSource, err := format.Source(rawSource)
	if err != nil {
		// Output raw source for debugging if formatting fails
		fmt.Printf("%s\n", rawSource)
		log.Fatal("error formatting generated code: ", err)
	}

	// Write to output file
	outputPath := "../../conditionalCmpConst_test.go"
	if err := os.WriteFile(outputPath, formattedSource, 0666); err != nil {
		log.Fatal("failed to write output file: ", err)
	}

	log.Printf("Tests successfully generated to %s", outputPath)
}
