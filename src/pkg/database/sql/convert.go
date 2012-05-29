// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Type conversions for Scan.

package sql

import (
	"database/sql/driver"
	"errors"
	"fmt"
	"reflect"
	"strconv"
)

// driverArgs converts arguments from callers of Stmt.Exec and
// Stmt.Query into driver Values.
//
// The statement si may be nil, if no statement is available.
func driverArgs(si driver.Stmt, args []interface{}) ([]driver.Value, error) {
	dargs := make([]driver.Value, len(args))
	cc, ok := si.(driver.ColumnConverter)

	// Normal path, for a driver.Stmt that is not a ColumnConverter.
	if !ok {
		for n, arg := range args {
			var err error
			dargs[n], err = driver.DefaultParameterConverter.ConvertValue(arg)
			if err != nil {
				return nil, fmt.Errorf("sql: converting Exec argument #%d's type: %v", n, err)
			}
		}
		return dargs, nil
	}

	// Let the Stmt convert its own arguments.
	for n, arg := range args {
		// First, see if the value itself knows how to convert
		// itself to a driver type.  For example, a NullString
		// struct changing into a string or nil.
		if svi, ok := arg.(driver.Valuer); ok {
			sv, err := svi.Value()
			if err != nil {
				return nil, fmt.Errorf("sql: argument index %d from Value: %v", n, err)
			}
			if !driver.IsValue(sv) {
				return nil, fmt.Errorf("sql: argument index %d: non-subset type %T returned from Value", n, sv)
			}
			arg = sv
		}

		// Second, ask the column to sanity check itself. For
		// example, drivers might use this to make sure that
		// an int64 values being inserted into a 16-bit
		// integer field is in range (before getting
		// truncated), or that a nil can't go into a NOT NULL
		// column before going across the network to get the
		// same error.
		var err error
		dargs[n], err = cc.ColumnConverter(n).ConvertValue(arg)
		if err != nil {
			return nil, fmt.Errorf("sql: converting argument #%d's type: %v", n, err)
		}
		if !driver.IsValue(dargs[n]) {
			return nil, fmt.Errorf("sql: driver ColumnConverter error converted %T to unsupported type %T",
				arg, dargs[n])
		}
	}

	return dargs, nil
}

// convertAssign copies to dest the value in src, converting it if possible.
// An error is returned if the copy would result in loss of information.
// dest should be a pointer type.
func convertAssign(dest, src interface{}) error {
	// Common cases, without reflect.  Fall through.
	switch s := src.(type) {
	case string:
		switch d := dest.(type) {
		case *string:
			*d = s
			return nil
		case *[]byte:
			*d = []byte(s)
			return nil
		}
	case []byte:
		switch d := dest.(type) {
		case *string:
			*d = string(s)
			return nil
		case *interface{}:
			bcopy := make([]byte, len(s))
			copy(bcopy, s)
			*d = bcopy
			return nil
		case *[]byte:
			*d = s
			return nil
		}
	case nil:
		switch d := dest.(type) {
		case *[]byte:
			*d = nil
			return nil
		}
	}

	var sv reflect.Value

	switch d := dest.(type) {
	case *string:
		sv = reflect.ValueOf(src)
		switch sv.Kind() {
		case reflect.Bool,
			reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
			reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
			reflect.Float32, reflect.Float64:
			*d = fmt.Sprintf("%v", src)
			return nil
		}
	case *bool:
		bv, err := driver.Bool.ConvertValue(src)
		if err == nil {
			*d = bv.(bool)
		}
		return err
	case *interface{}:
		*d = src
		return nil
	}

	if scanner, ok := dest.(Scanner); ok {
		return scanner.Scan(src)
	}

	dpv := reflect.ValueOf(dest)
	if dpv.Kind() != reflect.Ptr {
		return errors.New("destination not a pointer")
	}

	if !sv.IsValid() {
		sv = reflect.ValueOf(src)
	}

	dv := reflect.Indirect(dpv)
	if dv.Kind() == sv.Kind() {
		dv.Set(sv)
		return nil
	}

	switch dv.Kind() {
	case reflect.Ptr:
		if src == nil {
			dv.Set(reflect.Zero(dv.Type()))
			return nil
		} else {
			dv.Set(reflect.New(dv.Type().Elem()))
			return convertAssign(dv.Interface(), src)
		}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		s := asString(src)
		i64, err := strconv.ParseInt(s, 10, dv.Type().Bits())
		if err != nil {
			return fmt.Errorf("converting string %q to a %s: %v", s, dv.Kind(), err)
		}
		dv.SetInt(i64)
		return nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		s := asString(src)
		u64, err := strconv.ParseUint(s, 10, dv.Type().Bits())
		if err != nil {
			return fmt.Errorf("converting string %q to a %s: %v", s, dv.Kind(), err)
		}
		dv.SetUint(u64)
		return nil
	case reflect.Float32, reflect.Float64:
		s := asString(src)
		f64, err := strconv.ParseFloat(s, dv.Type().Bits())
		if err != nil {
			return fmt.Errorf("converting string %q to a %s: %v", s, dv.Kind(), err)
		}
		dv.SetFloat(f64)
		return nil
	}

	return fmt.Errorf("unsupported driver -> Scan pair: %T -> %T", src, dest)
}

func asString(src interface{}) string {
	switch v := src.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	}
	return fmt.Sprintf("%v", src)
}
