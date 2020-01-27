/*
 * Copyright 2020 bigpigeon. All rights reserved.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file.
 *
 */

package main

import (
	"go/ast"
	"go/token"
	"strings"
)

type tagFormatter struct {
	Err error
	fs  *token.FileSet
}

func (s *tagFormatter) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case *ast.StructType:
		if n.Fields != nil {
			var longestList []int
			var start int
			var end int
			var preFieldLine int
			for i, field := range n.Fields.List {
				line := s.fs.Position(field.Pos()).Line
				if preFieldLine+1 < line {
					fieldsTagFormat(n.Fields.List[start:end], longestList)
					start = i
					end = i
					longestList = nil
				}
				preFieldLine = line
				if field.Tag != nil {
					end = i
					_, keyValues, err := ParseTag(field.Tag.Value)
					if err != nil {
						s.Err = err
						return nil
					}
					for i, kv := range keyValues {
						if len(longestList) <= i {
							longestList = append(longestList, 0)
						}
						longestList[i] = max(len(kv.KeyValue), longestList[i])
					}
				} else {

				}
			}
			fieldsTagFormat(n.Fields.List[start:], longestList)
		}
	}
	return s
}

func fieldsTagFormat(fields []*ast.Field, longestList []int) {
	for _, f := range fields {
		if f.Tag != nil {
			quote, keyValues, err := ParseTag(f.Tag.Value)
			if err != nil {
				// must be nil error
				panic(err)
			}
			var keyValueRaw []string
			for i, kv := range keyValues {
				keyValueRaw = append(keyValueRaw, kv.KeyValue+strings.Repeat(" ", longestList[i]-len(kv.KeyValue)))
			}

			f.Tag.Value = quote + strings.TrimRight(strings.Join(keyValueRaw, " "), " ") + quote
			f.Tag.ValuePos = 0
		}
	}
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func tagFmt(f *ast.File, fs *token.FileSet) error {
	s := &tagFormatter{fs: fileSet}
	ast.Walk(s, f)
	return s.Err
}
