// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package xml

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
)

// typeInfo holds details for the xml representation of a type.
type typeInfo struct {
	xmlname *fieldInfo
	fields  []fieldInfo
}

// fieldInfo holds details for the xml representation of a single field.
type fieldInfo struct {
	idx     []int
	name    string
	xmlns   string
	flags   fieldFlags
	parents []string
}

type fieldFlags int

const (
	fElement fieldFlags = 1 << iota
	fAttr
	fCharData
	fInnerXml
	fComment
	fAny

	fOmitEmpty

	fMode = fElement | fAttr | fCharData | fInnerXml | fComment | fAny
)

var tinfoMap = make(map[reflect.Type]*typeInfo)
var tinfoLock sync.RWMutex

var nameType = reflect.TypeOf(Name{})

// getTypeInfo returns the typeInfo structure with details necessary
// for marshalling and unmarshalling typ.
func getTypeInfo(typ reflect.Type) (*typeInfo, error) {
	tinfoLock.RLock()
	tinfo, ok := tinfoMap[typ]
	tinfoLock.RUnlock()
	if ok {
		return tinfo, nil
	}
	tinfo = &typeInfo{}
	if typ.Kind() == reflect.Struct && typ != nameType {
		n := typ.NumField()
		for i := 0; i < n; i++ {
			f := typ.Field(i)
			if f.PkgPath != "" || f.Tag.Get("xml") == "-" {
				continue // Private field
			}

			// For embedded structs, embed its fields.
			if f.Anonymous {
				t := f.Type
				if t.Kind() == reflect.Ptr {
					t = t.Elem()
				}
				if t.Kind() == reflect.Struct {
					inner, err := getTypeInfo(t)
					if err != nil {
						return nil, err
					}
					if tinfo.xmlname == nil {
						tinfo.xmlname = inner.xmlname
					}
					for _, finfo := range inner.fields {
						finfo.idx = append([]int{i}, finfo.idx...)
						if err := addFieldInfo(typ, tinfo, &finfo); err != nil {
							return nil, err
						}
					}
					continue
				}
			}

			finfo, err := structFieldInfo(typ, &f)
			if err != nil {
				return nil, err
			}

			if f.Name == "XMLName" {
				tinfo.xmlname = finfo
				continue
			}

			// Add the field if it doesn't conflict with other fields.
			if err := addFieldInfo(typ, tinfo, finfo); err != nil {
				return nil, err
			}
		}
	}
	tinfoLock.Lock()
	tinfoMap[typ] = tinfo
	tinfoLock.Unlock()
	return tinfo, nil
}

// structFieldInfo builds and returns a fieldInfo for f.
func structFieldInfo(typ reflect.Type, f *reflect.StructField) (*fieldInfo, error) {
	finfo := &fieldInfo{idx: f.Index}

	// Split the tag from the xml namespace if necessary.
	tag := f.Tag.Get("xml")
	if i := strings.Index(tag, " "); i >= 0 {
		finfo.xmlns, tag = tag[:i], tag[i+1:]
	}

	// Parse flags.
	tokens := strings.Split(tag, ",")
	if len(tokens) == 1 {
		finfo.flags = fElement
	} else {
		tag = tokens[0]
		for _, flag := range tokens[1:] {
			switch flag {
			case "attr":
				finfo.flags |= fAttr
			case "chardata":
				finfo.flags |= fCharData
			case "innerxml":
				finfo.flags |= fInnerXml
			case "comment":
				finfo.flags |= fComment
			case "any":
				finfo.flags |= fAny
			case "omitempty":
				finfo.flags |= fOmitEmpty
			}
		}

		// Validate the flags used.
		valid := true
		switch mode := finfo.flags & fMode; mode {
		case 0:
			finfo.flags |= fElement
		case fAttr, fCharData, fInnerXml, fComment, fAny:
			if f.Name == "XMLName" || tag != "" && mode != fAttr {
				valid = false
			}
		default:
			// This will also catch multiple modes in a single field.
			valid = false
		}
		if finfo.flags&fMode == fAny {
			finfo.flags |= fElement
		}
		if finfo.flags&fOmitEmpty != 0 && finfo.flags&(fElement|fAttr) == 0 {
			valid = false
		}
		if !valid {
			return nil, fmt.Errorf("xml: invalid tag in field %s of type %s: %q",
				f.Name, typ, f.Tag.Get("xml"))
		}
	}

	// Use of xmlns without a name is not allowed.
	if finfo.xmlns != "" && tag == "" {
		return nil, fmt.Errorf("xml: namespace without name in field %s of type %s: %q",
			f.Name, typ, f.Tag.Get("xml"))
	}

	if f.Name == "XMLName" {
		// The XMLName field records the XML element name. Don't
		// process it as usual because its name should default to
		// empty rather than to the field name.
		finfo.name = tag
		return finfo, nil
	}

	if tag == "" {
		// If the name part of the tag is completely empty, get
		// default from XMLName of underlying struct if feasible,
		// or field name otherwise.
		if xmlname := lookupXMLName(f.Type); xmlname != nil {
			finfo.xmlns, finfo.name = xmlname.xmlns, xmlname.name
		} else {
			finfo.name = f.Name
		}
		return finfo, nil
	}

	// Prepare field name and parents.
	parents := strings.Split(tag, ">")
	if parents[0] == "" {
		parents[0] = f.Name
	}
	if parents[len(parents)-1] == "" {
		return nil, fmt.Errorf("xml: trailing '>' in field %s of type %s", f.Name, typ)
	}
	finfo.name = parents[len(parents)-1]
	if len(parents) > 1 {
		if (finfo.flags & fElement) == 0 {
			return nil, fmt.Errorf("xml: %s chain not valid with %s flag", tag, strings.Join(tokens[1:], ","))
		}
		finfo.parents = parents[:len(parents)-1]
	}

	// If the field type has an XMLName field, the names must match
	// so that the behavior of both marshalling and unmarshalling
	// is straightforward and unambiguous.
	if finfo.flags&fElement != 0 {
		ftyp := f.Type
		xmlname := lookupXMLName(ftyp)
		if xmlname != nil && xmlname.name != finfo.name {
			return nil, fmt.Errorf("xml: name %q in tag of %s.%s conflicts with name %q in %s.XMLName",
				finfo.name, typ, f.Name, xmlname.name, ftyp)
		}
	}
	return finfo, nil
}

// lookupXMLName returns the fieldInfo for typ's XMLName field
// in case it exists and has a valid xml field tag, otherwise
// it returns nil.
func lookupXMLName(typ reflect.Type) (xmlname *fieldInfo) {
	for typ.Kind() == reflect.Ptr {
		typ = typ.Elem()
	}
	if typ.Kind() != reflect.Struct {
		return nil
	}
	for i, n := 0, typ.NumField(); i < n; i++ {
		f := typ.Field(i)
		if f.Name != "XMLName" {
			continue
		}
		finfo, err := structFieldInfo(typ, &f)
		if finfo.name != "" && err == nil {
			return finfo
		}
		// Also consider errors as a non-existent field tag
		// and let getTypeInfo itself report the error.
		break
	}
	return nil
}

func min(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

// addFieldInfo adds finfo to tinfo.fields if there are no
// conflicts, or if conflicts arise from previous fields that were
// obtained from deeper embedded structures than finfo. In the latter
// case, the conflicting entries are dropped.
// A conflict occurs when the path (parent + name) to a field is
// itself a prefix of another path, or when two paths match exactly.
// It is okay for field paths to share a common, shorter prefix.
func addFieldInfo(typ reflect.Type, tinfo *typeInfo, newf *fieldInfo) error {
	var conflicts []int
Loop:
	// First, figure all conflicts. Most working code will have none.
	for i := range tinfo.fields {
		oldf := &tinfo.fields[i]
		if oldf.flags&fMode != newf.flags&fMode {
			continue
		}
		if oldf.xmlns != "" && newf.xmlns != "" && oldf.xmlns != newf.xmlns {
			continue
		}
		minl := min(len(newf.parents), len(oldf.parents))
		for p := 0; p < minl; p++ {
			if oldf.parents[p] != newf.parents[p] {
				continue Loop
			}
		}
		if len(oldf.parents) > len(newf.parents) {
			if oldf.parents[len(newf.parents)] == newf.name {
				conflicts = append(conflicts, i)
			}
		} else if len(oldf.parents) < len(newf.parents) {
			if newf.parents[len(oldf.parents)] == oldf.name {
				conflicts = append(conflicts, i)
			}
		} else {
			if newf.name == oldf.name {
				conflicts = append(conflicts, i)
			}
		}
	}
	// Without conflicts, add the new field and return.
	if conflicts == nil {
		tinfo.fields = append(tinfo.fields, *newf)
		return nil
	}

	// If any conflict is shallower, ignore the new field.
	// This matches the Go field resolution on embedding.
	for _, i := range conflicts {
		if len(tinfo.fields[i].idx) < len(newf.idx) {
			return nil
		}
	}

	// Otherwise, if any of them is at the same depth level, it's an error.
	for _, i := range conflicts {
		oldf := &tinfo.fields[i]
		if len(oldf.idx) == len(newf.idx) {
			f1 := typ.FieldByIndex(oldf.idx)
			f2 := typ.FieldByIndex(newf.idx)
			return &TagPathError{typ, f1.Name, f1.Tag.Get("xml"), f2.Name, f2.Tag.Get("xml")}
		}
	}

	// Otherwise, the new field is shallower, and thus takes precedence,
	// so drop the conflicting fields from tinfo and append the new one.
	for c := len(conflicts) - 1; c >= 0; c-- {
		i := conflicts[c]
		copy(tinfo.fields[i:], tinfo.fields[i+1:])
		tinfo.fields = tinfo.fields[:len(tinfo.fields)-1]
	}
	tinfo.fields = append(tinfo.fields, *newf)
	return nil
}

// A TagPathError represents an error in the unmarshalling process
// caused by the use of field tags with conflicting paths.
type TagPathError struct {
	Struct       reflect.Type
	Field1, Tag1 string
	Field2, Tag2 string
}

func (e *TagPathError) Error() string {
	return fmt.Sprintf("%s field %q with tag %q conflicts with field %q with tag %q", e.Struct, e.Field1, e.Tag1, e.Field2, e.Tag2)
}

// value returns v's field value corresponding to finfo.
// It's equivalent to v.FieldByIndex(finfo.idx), but initializes
// and dereferences pointers as necessary.
func (finfo *fieldInfo) value(v reflect.Value) reflect.Value {
	for i, x := range finfo.idx {
		if i > 0 {
			t := v.Type()
			if t.Kind() == reflect.Ptr && t.Elem().Kind() == reflect.Struct {
				if v.IsNil() {
					v.Set(reflect.New(v.Type().Elem()))
				}
				v = v.Elem()
			}
		}
		v = v.Field(x)
	}
	return v
}
