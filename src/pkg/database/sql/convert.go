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

// subsetTypeArgs takes a slice of arguments from callers of the sql
// package and converts them into a slice of the driver package's
// "subset types".
func subsetTypeArgs(args []interface{}) ([]interface{}, error) {
	out := make([]interface{}, len(args))
	for n, arg := range args {
		var err error
		out[n], err = driver.DefaultParameterConverter.ConvertValue(arg)
		if err != nil {
			return nil, fmt.Errorf("sql: converting argument #%d's type: %v", n+1, err)
		}
	}
	return out, nil
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
		}
	case []byte:
		switch d := dest.(type) {
		case *string:
			*d = string(s)
			return nil
		case *[]byte:
			*d = s
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
	}

	if scanner, ok := dest.(ScannerInto); ok {
		return scanner.ScanInto(src)
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
