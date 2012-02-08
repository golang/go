// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package driver

import (
	"fmt"
	"reflect"
	"strconv"
	"time"
)

// ValueConverter is the interface providing the ConvertValue method.
//
// Various implementations of ValueConverter are provided by the
// driver package to provide consistent implementations of conversions
// between drivers.  The ValueConverters have several uses:
//
//  * converting from the subset types as provided by the sql package
//    into a database table's specific column type and making sure it
//    fits, such as making sure a particular int64 fits in a
//    table's uint16 column.
//
//  * converting a value as given from the database into one of the
//    subset types.
//
//  * by the sql package, for converting from a driver's subset type
//    to a user's type in a scan.
type ValueConverter interface {
	// ConvertValue converts a value to a restricted subset type.
	ConvertValue(v interface{}) (interface{}, error)
}

// SubsetValuer is the interface providing the SubsetValue method.
//
// Types implementing SubsetValuer interface are able to convert
// themselves to one of the driver's allowed subset values.
type SubsetValuer interface {
	// SubsetValue returns a driver parameter subset value.
	SubsetValue() (interface{}, error)
}

// Bool is a ValueConverter that converts input values to bools.
//
// The conversion rules are:
//  - booleans are returned unchanged
//  - for integer types,
//       1 is true
//       0 is false,
//       other integers are an error
//  - for strings and []byte, same rules as strconv.ParseBool
//  - all other types are an error
var Bool boolType

type boolType struct{}

var _ ValueConverter = boolType{}

func (boolType) String() string { return "Bool" }

func (boolType) ConvertValue(src interface{}) (interface{}, error) {
	switch s := src.(type) {
	case bool:
		return s, nil
	case string:
		b, err := strconv.ParseBool(s)
		if err != nil {
			return nil, fmt.Errorf("sql/driver: couldn't convert %q into type bool", s)
		}
		return b, nil
	case []byte:
		b, err := strconv.ParseBool(string(s))
		if err != nil {
			return nil, fmt.Errorf("sql/driver: couldn't convert %q into type bool", s)
		}
		return b, nil
	}

	sv := reflect.ValueOf(src)
	switch sv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		iv := sv.Int()
		if iv == 1 || iv == 0 {
			return iv == 1, nil
		}
		return nil, fmt.Errorf("sql/driver: couldn't convert %d into type bool", iv)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		uv := sv.Uint()
		if uv == 1 || uv == 0 {
			return uv == 1, nil
		}
		return nil, fmt.Errorf("sql/driver: couldn't convert %d into type bool", uv)
	}

	return nil, fmt.Errorf("sql/driver: couldn't convert %v (%T) into type bool", src, src)
}

// Int32 is a ValueConverter that converts input values to int64,
// respecting the limits of an int32 value.
var Int32 int32Type

type int32Type struct{}

var _ ValueConverter = int32Type{}

func (int32Type) ConvertValue(v interface{}) (interface{}, error) {
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

func (stringType) ConvertValue(v interface{}) (interface{}, error) {
	switch v.(type) {
	case string, []byte:
		return v, nil
	}
	return fmt.Sprintf("%v", v), nil
}

// Null is a type that implements ValueConverter by allowing nil
// values but otherwise delegating to another ValueConverter.
type Null struct {
	Converter ValueConverter
}

func (n Null) ConvertValue(v interface{}) (interface{}, error) {
	if v == nil {
		return nil, nil
	}
	return n.Converter.ConvertValue(v)
}

// NotNull is a type that implements ValueConverter by disallowing nil
// values but otherwise delegating to another ValueConverter.
type NotNull struct {
	Converter ValueConverter
}

func (n NotNull) ConvertValue(v interface{}) (interface{}, error) {
	if v == nil {
		return nil, fmt.Errorf("nil value not allowed")
	}
	return n.Converter.ConvertValue(v)
}

// IsParameterSubsetType reports whether v is of a valid type for a
// parameter. These types are:
//
//   int64
//   float64
//   bool
//   nil
//   []byte
//   time.Time
//   string
//
// This is the same list as IsScanSubsetType, with the addition of
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
//   time.Time
//
// This is the same list as IsParameterSubsetType, without string.
func IsScanSubsetType(v interface{}) bool {
	if v == nil {
		return true
	}
	switch v.(type) {
	case int64, float64, []byte, bool, time.Time:
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

func (defaultConverter) ConvertValue(v interface{}) (interface{}, error) {
	if IsParameterSubsetType(v) {
		return v, nil
	}

	if svi, ok := v.(SubsetValuer); ok {
		sv, err := svi.SubsetValue()
		if err != nil {
			return nil, err
		}
		if !IsParameterSubsetType(sv) {
			return nil, fmt.Errorf("non-subset type %T returned from SubsetValue", sv)
		}
		return sv, nil
	}

	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Ptr:
		// indirect pointers
		if rv.IsNil() {
			return nil, nil
		} else {
			return defaultConverter{}.ConvertValue(rv.Elem().Interface())
		}
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
	return nil, fmt.Errorf("unsupported type %T, a %s", v, rv.Kind())
}
