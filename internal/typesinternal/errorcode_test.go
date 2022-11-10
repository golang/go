// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal_test

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/parser"
	"go/token"
	"go/types"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"testing"
)

func TestErrorCodes(t *testing.T) {
	t.Skip("unskip this test to verify the correctness of errorcode.go for the current Go version")

	// For older go versions, this file was src/go/types/errorcodes.go.
	stdPath := filepath.Join(runtime.GOROOT(), "src", "internal", "types", "errors", "codes.go")
	stdCodes, err := loadCodes(stdPath)
	if err != nil {
		t.Fatalf("loading std codes: %v", err)
	}

	localPath := "errorcode.go"
	localCodes, err := loadCodes(localPath)
	if err != nil {
		t.Fatalf("loading local codes: %v", err)
	}

	// Verify that all std codes are present, with the correct value.
	type codeVal struct {
		Name  string
		Value int64
	}
	var byValue []codeVal
	for k, v := range stdCodes {
		byValue = append(byValue, codeVal{k, v})
	}
	sort.Slice(byValue, func(i, j int) bool {
		return byValue[i].Value < byValue[j].Value
	})

	localLookup := make(map[int64]string)
	for k, v := range localCodes {
		if _, ok := localLookup[v]; ok {
			t.Errorf("duplicate error code value %d", v)
		}
		localLookup[v] = k
	}

	for _, std := range byValue {
		local, ok := localCodes[std.Name]
		if !ok {
			if v, ok := localLookup[std.Value]; ok {
				t.Errorf("Missing code for %s (code %d is %s)", std.Name, std.Value, v)
			} else {
				t.Errorf("Missing code for %s", std.Name)
			}
		}
		if local != std.Value {
			t.Errorf("Mismatching value for %s: got %d, but stdlib has %d", std.Name, local, std.Value)
		}
	}
}

// loadCodes loads all constant values found in filepath.
//
// The given file must type-check cleanly as a standalone file.
func loadCodes(filepath string) (map[string]int64, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, filepath, nil, 0)
	if err != nil {
		return nil, err
	}
	var config types.Config
	pkg, err := config.Check("p", fset, []*ast.File{f}, nil)
	if err != nil {
		return nil, err
	}

	codes := make(map[string]int64)
	for _, name := range pkg.Scope().Names() {
		obj := pkg.Scope().Lookup(name)
		c, ok := obj.(*types.Const)
		if !ok {
			continue
		}
		name := strings.TrimPrefix(name, "_") // compatibility with earlier go versions
		codes[name], ok = constant.Int64Val(c.Val())
		if !ok {
			return nil, fmt.Errorf("non integral value %v for %s", c.Val(), name)
		}
	}
	if len(codes) < 100 {
		return nil, fmt.Errorf("sanity check: got %d codes but expected at least 100", len(codes))
	}
	return codes, nil
}
