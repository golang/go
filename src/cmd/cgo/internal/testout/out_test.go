// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package out_test

import (
	"bufio"
	"bytes"
	"fmt"
	"internal/goarch"
	"internal/testenv"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

type methodAlign struct {
	Method string
	Align  int
}

var wantAligns = map[string]int{
	"ReturnEmpty":         1,
	"ReturnOnlyUint8":     1,
	"ReturnOnlyUint16":    2,
	"ReturnOnlyUint32":    4,
	"ReturnOnlyUint64":    goarch.PtrSize,
	"ReturnOnlyInt":       goarch.PtrSize,
	"ReturnOnlyPtr":       goarch.PtrSize,
	"ReturnByteSlice":     goarch.PtrSize,
	"ReturnString":        goarch.PtrSize,
	"InputAndReturnUint8": 1,
	"MixedTypes":          goarch.PtrSize,
}

// TestAligned tests that the generated _cgo_export.c file has the wanted
// align attributes for struct types used as arguments or results of
// //exported functions.
func TestAligned(t *testing.T) {
	testenv.MustHaveGoRun(t)
	testenv.MustHaveCGO(t)

	testdata, err := filepath.Abs("testdata")
	if err != nil {
		t.Fatal(err)
	}

	objDir := t.TempDir()

	cmd := testenv.Command(t, testenv.GoToolPath(t), "tool", "cgo",
		"-objdir", objDir,
		filepath.Join(testdata, "aligned.go"))
	cmd.Stderr = new(bytes.Buffer)

	err = cmd.Run()
	if err != nil {
		t.Fatalf("%#q: %v\n%s", cmd, err, cmd.Stderr)
	}

	haveAligns, err := parseAlign(filepath.Join(objDir, "_cgo_export.c"))
	if err != nil {
		t.Fatal(err)
	}

	// Check that we have all the wanted methods
	if len(haveAligns) != len(wantAligns) {
		t.Fatalf("have %d methods with aligned, want %d", len(haveAligns), len(wantAligns))
	}

	for i := range haveAligns {
		method := haveAligns[i].Method
		haveAlign := haveAligns[i].Align

		wantAlign, ok := wantAligns[method]
		if !ok {
			t.Errorf("method %s: have aligned %d, want missing entry", method, haveAlign)
		} else if haveAlign != wantAlign {
			t.Errorf("method %s: have aligned %d, want %d", method, haveAlign, wantAlign)
		}
	}
}

func parseAlign(filename string) ([]methodAlign, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	var results []methodAlign
	scanner := bufio.NewScanner(file)

	// Regex to match function declarations like "struct MethodName_return MethodName("
	funcRegex := regexp.MustCompile(`^struct\s+(\w+)_return\s+(\w+)\(`)
	// Regex to match simple function declarations like "GoSlice MethodName("
	simpleFuncRegex := regexp.MustCompile(`^Go\w+\s+(\w+)\(`)
	// Regex to match void-returning exported functions like "void ReturnEmpty("
	voidFuncRegex := regexp.MustCompile(`^void\s+(\w+)\(`)
	// Regex to match align attributes like "__attribute__((aligned(8)))"
	alignRegex := regexp.MustCompile(`__attribute__\(\(aligned\((\d+)\)\)\)`)

	var currentMethod string

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())

		// Check if this line declares a function with struct return type
		if matches := funcRegex.FindStringSubmatch(line); matches != nil {
			currentMethod = matches[2] // Extract the method name
		} else if matches := simpleFuncRegex.FindStringSubmatch(line); matches != nil {
			// Check if this line declares a function with simple return type (like GoSlice)
			currentMethod = matches[1] // Extract the method name
		} else if matches := voidFuncRegex.FindStringSubmatch(line); matches != nil {
			// Check if this line declares a void-returning function
			currentMethod = matches[1] // Extract the method name
		}

		// Check if this line contains align information
		if alignMatches := alignRegex.FindStringSubmatch(line); alignMatches != nil && currentMethod != "" {
			alignStr := alignMatches[1]
			align, err := strconv.Atoi(alignStr)
			if err != nil {
				// Skip this entry if we can't parse the align as integer
				currentMethod = ""
				continue
			}
			results = append(results, methodAlign{
				Method: currentMethod,
				Align:  align,
			})
			currentMethod = "" // Reset for next method
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	return results, nil
}
