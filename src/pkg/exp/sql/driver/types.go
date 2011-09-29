// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package driver

import (
	"fmt"
	"os"
	"reflect"
	"strconv"
)

// ValueConverter is the interface providing the ConvertValue method.
type ValueConverter interface {
	// ConvertValue converts a value to a restricted subset type.
	ConvertValue(v interface{}) (interface{}, os.Error)
}

// Bool is a ValueConverter that converts input values to bools.
//
// The conversion rules are:
//  - .... TODO(bradfitz): TBD
var Bool boolType

type boolType struct{}

var _ ValueConverter = boolType{}

func (boolType) ConvertValue(v interface{}) (interface{}, os.Error) {
	return nil, fmt.Errorf("TODO(bradfitz): bool conversions")
}

// Int32 is a ValueConverter that converts input values to int64,
// respecting the limits of an int32 value.
var Int32 int32Type

type int32Type struct{}

var _ ValueConverter = int32Type{}

func (int32Type) ConvertValue(v interface{}) (interface{}, os.Error) {
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		i64 := rv.Int()
		if i64 > (1<<31)-1 || i64 < -(1<<31) {
			return nil, fmt.Errorf("sql/driver: value %d overflows int32", v)
		}
		return i64, nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		u64 := rv.Uint()
		if u64 > (1<<31)-1 {
			return nil, fmt.Errorf("sql/driver: value %d overflows int32", v)
		}
		return int64(u64), nil
	case reflect.String:
		i, err := strconv.Atoi(rv.String())
		if err != nil {
			return nil, fmt.Errorf("sql/driver: value %q can't be converted to int32", v)
		}
		return int64(i), nil
	}
	return nil, fmt.Errorf("sql/driver: unsupported value %v (type %T) converting to int32", v, v)
}

// String is a ValueConverter that converts its input to a string.
// If the value is already a string or []byte, it's unchanged.
// If the value is of another type, conversion to string is done
// with fmt.Sprintf("%v", v).
var String stringType

type stringType struct{}

func (stringType) ConvertValue(v interface{}) (interface{}, os.Error) {
	switch v.(type) {
	case string, []byte:
		return v, nil
	}
	return fmt.Sprintf("%v", v), nil
}

// IsParameterSubsetType reports whether v is of a valid type for a
// parameter. These types are:
//
//   int64
//   float64
//   bool
//   nil
//   []byte
//   string
//
// This is the ame list as IsScanSubsetType, with the addition of
// string.
func IsParameterSubsetType(v interface{}) bool {
	if IsScanSubsetType(v) {
		return true
	}
	if _, ok := v.(string); ok {
		return true
	}
	return false
}

// IsScanSubsetType reports whether v is of a valid type for a
// value populated by Rows.Next. These types are:
//
//   int64
//   float64
//   bool
//   nil
//   []byte
//
// This is the same list as IsParameterSubsetType, without string.
func IsScanSubsetType(v interface{}) bool {
	if v == nil {
		return true
	}
	switch v.(type) {
	case int64, float64, []byte, bool:
		return true
	}
	return false
}

// DefaultParameterConverter is the default implementation of
// ValueConverter that's used when a Stmt doesn't implement
// ColumnConverter.
//
// DefaultParameterConverter returns the given value directly if
// IsSubsetType(value).  Otherwise integer type are converted to
// int64, floats to float64, and strings to []byte.  Other types are
// an error.
var DefaultParameterConverter defaultConverter

type defaultConverter struct{}

var _ ValueConverter = defaultConverter{}

func (defaultConverter) ConvertValue(v interface{}) (interface{}, os.Error) {
	if IsParameterSubsetType(v) {
		return v, nil
	}

	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return rv.Int(), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32:
		return int64(rv.Uint()), nil
	case reflect.Uint64:
		u64 := rv.Uint()
		if u64 >= 1<<63 {
			return nil, fmt.Errorf("uint64 values with high bit set are not supported")
		}
		return int64(u64), nil
	case reflect.Float32, reflect.Float64:
		return rv.Float(), nil
	}
	return nil, fmt.Errorf("unsupported type %s", rv.Kind())
}
