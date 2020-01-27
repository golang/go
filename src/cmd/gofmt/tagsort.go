/*
 * Copyright 2020 bigpigeon. All rights reserved.
 * Use of this source code is governed by a MIT style
 * license that can be found in the LICENSE file.
 *
 */

package main

import (
	"errors"
	"go/ast"
	"sort"
	"strings"
)

var ErrInvalidTag = errors.New("Invalid tag ")

type tagSorter struct {
	Err error
}

func (s *tagSorter) Visit(node ast.Node) ast.Visitor {
	switch n := node.(type) {
	case *ast.StructType:
		if n.Fields != nil {

			for _, field := range n.Fields.List {
				if field.Tag != nil {

					quote, keyValues, err := ParseTag(field.Tag.Value)
					if err != nil {
						s.Err = err
						return nil
					}
					sort.Slice(keyValues, func(i, j int) bool {
						return keyValues[i].Key < keyValues[j].Key
					})
					var keyValuesRaw []string
					for _, kv := range keyValues {
						keyValuesRaw = append(keyValuesRaw, kv.KeyValue)
					}

					field.Tag.Value = quote + strings.Join(keyValuesRaw, " ") + quote
					field.Tag.ValuePos = 0

				}
			}
		}
	}
	return s
}

func tagSortByKey(f *ast.File) error {
	s := &tagSorter{}
	ast.Walk(s, f)
	return s.Err
}

type KeyValue struct {
	Key      string
	KeyValue string
}

// ParseTag returns all tag keys and tags key:"Value" list
func ParseTag(tag string) (quote string, keyValues []KeyValue, err error) {
	if len(tag) < 2 {
		err = ErrInvalidTag
		return
	}
	quote = tag[:1]
	tag = tag[1 : len(tag)-1]

	for tag != "" {
		// Skip leading space.
		i := 0
		for i < len(tag) && tag[i] == ' ' {
			i++
		}
		tag = tag[i:]
		if tag == "" {
			break
		}

		// Scan to colon. A space, a quote or a control character is a syntax error.
		// Strictly speaking, control chars include the range [0x7f, 0x9f], not just
		// [0x00, 0x1f], but in practice, we ignore the multi-byte control characters
		// as it is simpler to inspect the tag's bytes than the tag's runes.
		i = 0
		for i < len(tag) && tag[i] > ' ' && tag[i] != ':' && tag[i] != '"' && tag[i] != 0x7f {
			i++
		}
		if i == 0 || i+1 >= len(tag) || tag[i] != ':' || tag[i+1] != '"' {
			break
		}
		name := string(tag[:i])

		// Scan quoted string to find value.
		i = i + 2
		for i < len(tag) && tag[i] != '"' {
			if tag[i] == '\\' {
				i++
			}
			i++
		}
		if i >= len(tag) {
			return "", nil, ErrInvalidTag
		}
		keyValue := string(tag[:i+1])
		keyValues = append(keyValues, KeyValue{
			Key:      name,
			KeyValue: keyValue,
		})

		tag = tag[i+1:]
	}
	return
}
