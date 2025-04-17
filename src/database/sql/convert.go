// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Type conversions for Scan.

package sql

import (
	"bytes"
	"database/sql/driver"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"time"
	"unicode"
	"unicode/utf8"
	_ "unsafe" // for linkname
)

var errNilPtr = errors.New("destination pointer is nil") // embedded in descriptive error

func describeNamedValue(nv *driver.NamedValue) string {
	if len(nv.Name) == 0 {
		return fmt.Sprintf("$%d", nv.Ordinal)
	}
	return fmt.Sprintf("with name %q", nv.Name)
}

func validateNamedValueName(name string) error {
	if len(name) == 0 {
		return nil
	}
	r, _ := utf8.DecodeRuneInString(name)
	if unicode.IsLetter(r) {
		return nil
	}
	return fmt.Errorf("name %q does not begin with a letter", name)
}

// ccChecker wraps the driver.ColumnConverter and allows it to be used
// as if it were a NamedValueChecker. If the driver ColumnConverter
// is not present then the NamedValueChecker will return driver.ErrSkip.
type ccChecker struct {
	cci  driver.ColumnConverter
	want int
}

func (c ccChecker) CheckNamedValue(nv *driver.NamedValue) error {
	if c.cci == nil {
		return driver.ErrSkip
	}
	// The column converter shouldn't be called on any index
	// it isn't expecting. The final error will be thrown
	// in the argument converter loop.
	index := nv.Ordinal - 1
	if c.want <= index {
		return nil
	}

	// First, see if the value itself knows how to convert
	// itself to a driver type. For example, a NullString
	// struct changing into a string or nil.
	if vr, ok := nv.Value.(driver.Valuer); ok {
		sv, err := callValuerValue(vr)
		if err != nil {
			return err
		}
		if !driver.IsValue(sv) {
			return fmt.Errorf("non-subset type %T returned from Value", sv)
		}
		nv.Value = sv
	}

	// Second, ask the column to sanity check itself. For
	// example, drivers might use this to make sure that
	// an int64 values being inserted into a 16-bit
	// integer field is in range (before getting
	// truncated), or that a nil can't go into a NOT NULL
	// column before going across the network to get the
	// same error.
	var err error
	arg := nv.Value
	nv.Value, err = c.cci.ColumnConverter(index).ConvertValue(arg)
	if err != nil {
		return err
	}
	if !driver.IsValue(nv.Value) {
		return fmt.Errorf("driver ColumnConverter error converted %T to unsupported type %T", arg, nv.Value)
	}
	return nil
}

// defaultCheckNamedValue wraps the default ColumnConverter to have the same
// function signature as the CheckNamedValue in the driver.NamedValueChecker
// interface.
func defaultCheckNamedValue(nv *driver.NamedValue) (err error) {
	nv.Value, err = driver.DefaultParameterConverter.ConvertValue(nv.Value)
	return err
}

// driverArgsConnLocked converts arguments from callers of Stmt.Exec and
// Stmt.Query into driver Values.
//
// The statement ds may be nil, if no statement is available.
//
// ci must be locked.
func driverArgsConnLocked(ci driver.Conn, ds *driverStmt, args []any) ([]driver.NamedValue, error) {
	nvargs := make([]driver.NamedValue, len(args))

	// -1 means the driver doesn't know how to count the number of
	// placeholders, so we won't sanity check input here and instead let the
	// driver deal with errors.
	want := -1

	var si driver.Stmt
	var cc ccChecker
	if ds != nil {
		si = ds.si
		want = ds.si.NumInput()
		cc.want = want
	}

	// Check all types of interfaces from the start.
	// Drivers may opt to use the NamedValueChecker for special
	// argument types, then return driver.ErrSkip to pass it along
	// to the column converter.
	nvc, ok := si.(driver.NamedValueChecker)
	if !ok {
		nvc, _ = ci.(driver.NamedValueChecker)
	}
	cci, ok := si.(driver.ColumnConverter)
	if ok {
		cc.cci = cci
	}

	// Loop through all the arguments, checking each one.
	// If no error is returned simply increment the index
	// and continue. However, if driver.ErrRemoveArgument
	// is returned the argument is not included in the query
	// argument list.
	var err error
	var n int
	for _, arg := range args {
		nv := &nvargs[n]
		if np, ok := arg.(NamedArg); ok {
			if err = validateNamedValueName(np.Name); err != nil {
				return nil, err
			}
			arg = np.Value
			nv.Name = np.Name
		}
		nv.Ordinal = n + 1
		nv.Value = arg

		// Checking sequence has four routes:
		// A: 1. Default
		// B: 1. NamedValueChecker 2. Column Converter 3. Default
		// C: 1. NamedValueChecker 3. Default
		// D: 1. Column Converter 2. Default
		//
		// The only time a Column Converter is called is first
		// or after NamedValueConverter. If first it is handled before
		// the nextCheck label. Thus for repeats tries only when the
		// NamedValueConverter is selected should the Column Converter
		// be used in the retry.
		checker := defaultCheckNamedValue
		nextCC := false
		switch {
		case nvc != nil:
			nextCC = cci != nil
			checker = nvc.CheckNamedValue
		case cci != nil:
			checker = cc.CheckNamedValue
		}

	nextCheck:
		err = checker(nv)
		switch err {
		case nil:
			n++
			continue
		case driver.ErrRemoveArgument:
			nvargs = nvargs[:len(nvargs)-1]
			continue
		case driver.ErrSkip:
			if nextCC {
				nextCC = false
				checker = cc.CheckNamedValue
			} else {
				checker = defaultCheckNamedValue
			}
			goto nextCheck
		default:
			return nil, fmt.Errorf("sql: converting argument %s type: %w", describeNamedValue(nv), err)
		}
	}

	// Check the length of arguments after conversion to allow for omitted
	// arguments.
	if want != -1 && len(nvargs) != want {
		return nil, fmt.Errorf("sql: expected %d arguments, got %d", want, len(nvargs))
	}

	return nvargs, nil
}

// convertAssign is the same as convertAssignRows, but without the optional
// rows argument.
//
// convertAssign should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - ariga.io/entcache
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname convertAssign
func convertAssign(dest, src any) error {
	return convertAssignRows(dest, src, nil)
}

// convertAssignRows copies to dest the value in src, converting it if possible.
// An error is returned if the copy would result in loss of information.
// dest should be a pointer type. If rows is passed in, the rows will
// be used as the parent for any cursor values converted from a
// driver.Rows to a *Rows.
func convertAssignRows(dest, src any, rows *Rows) error {
	// Common cases, without reflect.
	switch s := src.(type) {
	case string:
		switch d := dest.(type) {
		case *string:
			if d == nil {
				return errNilPtr
			}
			*d = s
			return nil
		case *[]byte:
			if d == nil {
				return errNilPtr
			}
			*d = []byte(s)
			return nil
		case *RawBytes:
			if d == nil {
				return errNilPtr
			}
			*d = rows.setrawbuf(append(rows.rawbuf(), s...))
			return nil
		}
	case []byte:
		switch d := dest.(type) {
		case *string:
			if d == nil {
				return errNilPtr
			}
			*d = string(s)
			return nil
		case *any:
			if d == nil {
				return errNilPtr
			}
			*d = bytes.Clone(s)
			return nil
		case *[]byte:
			if d == nil {
				return errNilPtr
			}
			*d = bytes.Clone(s)
			return nil
		case *RawBytes:
			if d == nil {
				return errNilPtr
			}
			*d = s
			return nil
		}
	case time.Time:
		switch d := dest.(type) {
		case *time.Time:
			*d = s
			return nil
		case *string:
			*d = s.Format(time.RFC3339Nano)
			return nil
		case *[]byte:
			if d == nil {
				return errNilPtr
			}
			*d = s.AppendFormat(make([]byte, 0, len(time.RFC3339Nano)), time.RFC3339Nano)
			return nil
		case *RawBytes:
			if d == nil {
				return errNilPtr
			}
			*d = rows.setrawbuf(s.AppendFormat(rows.rawbuf(), time.RFC3339Nano))
			return nil
		}
	case decimalDecompose:
		switch d := dest.(type) {
		case decimalCompose:
			return d.Compose(s.Decompose(nil))
		}
	case nil:
		switch d := dest.(type) {
		case *any:
			if d == nil {
				return errNilPtr
			}
			*d = nil
			return nil
		case *[]byte:
			if d == nil {
				return errNilPtr
			}
			*d = nil
			return nil
		case *RawBytes:
			if d == nil {
				return errNilPtr
			}
			*d = nil
			return nil
		}
	// The driver is returning a cursor the client may iterate over.
	case driver.Rows:
		switch d := dest.(type) {
		case *Rows:
			if d == nil {
				return errNilPtr
			}
			if rows == nil {
				return errors.New("invalid context to convert cursor rows, missing parent *Rows")
			}
			rows.closemu.Lock()
			*d = Rows{
				dc:          rows.dc,
				releaseConn: func(error) {},
				rowsi:       s,
			}
			// Chain the cancel function.
			parentCancel := rows.cancel
			rows.cancel = func() {
				// When Rows.cancel is called, the closemu will be locked as well.
				// So we can access rs.lasterr.
				d.close(rows.lasterr)
				if parentCancel != nil {
					parentCancel()
				}
			}
			rows.closemu.Unlock()
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
			*d = asString(src)
			return nil
		}
	case *[]byte:
		sv = reflect.ValueOf(src)
		if b, ok := asBytes(nil, sv); ok {
			*d = b
			return nil
		}
	case *RawBytes:
		sv = reflect.ValueOf(src)
		if b, ok := asBytes(rows.rawbuf(), sv); ok {
			*d = rows.setrawbuf(b)
			return nil
		}
	case *bool:
		bv, err := driver.Bool.ConvertValue(src)
		if err == nil {
			*d = bv.(bool)
		}
		return err
	case *any:
		*d = src
		return nil
	}

	if scanner, ok := dest.(Scanner); ok {
		return scanner.Scan(src)
	}

	dpv := reflect.ValueOf(dest)
	if dpv.Kind() != reflect.Pointer {
		return errors.New("destination not a pointer")
	}
	if dpv.IsNil() {
		return errNilPtr
	}

	if !sv.IsValid() {
		sv = reflect.ValueOf(src)
	}

	dv := reflect.Indirect(dpv)
	if sv.IsValid() && sv.Type().AssignableTo(dv.Type()) {
		switch b := src.(type) {
		case []byte:
			dv.Set(reflect.ValueOf(bytes.Clone(b)))
		default:
			dv.Set(sv)
		}
		return nil
	}

	if dv.Kind() == sv.Kind() && sv.Type().ConvertibleTo(dv.Type()) {
		dv.Set(sv.Convert(dv.Type()))
		return nil
	}

	// The following conversions use a string value as an intermediate representation
	// to convert between various numeric types.
	//
	// This also allows scanning into user defined types such as "type Int int64".
	// For symmetry, also check for string destination types.
	switch dv.Kind() {
	case reflect.Pointer:
		if src == nil {
			dv.SetZero()
			return nil
		}
		dv.Set(reflect.New(dv.Type().Elem()))
		return convertAssignRows(dv.Interface(), src, rows)
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		if src == nil {
			return fmt.Errorf("converting NULL to %s is unsupported", dv.Kind())
		}
		s := asString(src)
		i64, err := strconv.ParseInt(s, 10, dv.Type().Bits())
		if err != nil {
			err = strconvErr(err)
			return fmt.Errorf("converting driver.Value type %T (%q) to a %s: %v", src, s, dv.Kind(), err)
		}
		dv.SetInt(i64)
		return nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		if src == nil {
			return fmt.Errorf("converting NULL to %s is unsupported", dv.Kind())
		}
		s := asString(src)
		u64, err := strconv.ParseUint(s, 10, dv.Type().Bits())
		if err != nil {
			err = strconvErr(err)
			return fmt.Errorf("converting driver.Value type %T (%q) to a %s: %v", src, s, dv.Kind(), err)
		}
		dv.SetUint(u64)
		return nil
	case reflect.Float32, reflect.Float64:
		if src == nil {
			return fmt.Errorf("converting NULL to %s is unsupported", dv.Kind())
		}
		s := asString(src)
		f64, err := strconv.ParseFloat(s, dv.Type().Bits())
		if err != nil {
			err = strconvErr(err)
			return fmt.Errorf("converting driver.Value type %T (%q) to a %s: %v", src, s, dv.Kind(), err)
		}
		dv.SetFloat(f64)
		return nil
	case reflect.String:
		if src == nil {
			return fmt.Errorf("converting NULL to %s is unsupported", dv.Kind())
		}
		switch v := src.(type) {
		case string:
			dv.SetString(v)
			return nil
		case []byte:
			dv.SetString(string(v))
			return nil
		}
	}

	return fmt.Errorf("unsupported Scan, storing driver.Value type %T into type %T", src, dest)
}

func strconvErr(err error) error {
	if ne, ok := err.(*strconv.NumError); ok {
		return ne.Err
	}
	return err
}

func asString(src any) string {
	switch v := src.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	}
	rv := reflect.ValueOf(src)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return strconv.FormatInt(rv.Int(), 10)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return strconv.FormatUint(rv.Uint(), 10)
	case reflect.Float64:
		return strconv.FormatFloat(rv.Float(), 'g', -1, 64)
	case reflect.Float32:
		return strconv.FormatFloat(rv.Float(), 'g', -1, 32)
	case reflect.Bool:
		return strconv.FormatBool(rv.Bool())
	}
	return fmt.Sprintf("%v", src)
}

func asBytes(buf []byte, rv reflect.Value) (b []byte, ok bool) {
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return strconv.AppendInt(buf, rv.Int(), 10), true
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return strconv.AppendUint(buf, rv.Uint(), 10), true
	case reflect.Float32:
		return strconv.AppendFloat(buf, rv.Float(), 'g', -1, 32), true
	case reflect.Float64:
		return strconv.AppendFloat(buf, rv.Float(), 'g', -1, 64), true
	case reflect.Bool:
		return strconv.AppendBool(buf, rv.Bool()), true
	case reflect.String:
		s := rv.String()
		return append(buf, s...), true
	}
	return
}

var valuerReflectType = reflect.TypeFor[driver.Valuer]()

// callValuerValue returns vr.Value(), with one exception:
// If vr.Value is an auto-generated method on a pointer type and the
// pointer is nil, it would panic at runtime in the panicwrap
// method. Treat it like nil instead.
// Issue 8415.
//
// This is so people can implement driver.Value on value types and
// still use nil pointers to those types to mean nil/NULL, just like
// string/*string.
//
// This function is mirrored in the database/sql/driver package.
func callValuerValue(vr driver.Valuer) (v driver.Value, err error) {
	if rv := reflect.ValueOf(vr); rv.Kind() == reflect.Pointer &&
		rv.IsNil() &&
		rv.Type().Elem().Implements(valuerReflectType) {
		return nil, nil
	}
	return vr.Value()
}

// decimal composes or decomposes a decimal value to and from individual parts.
// There are four parts: a boolean negative flag, a form byte with three possible states
// (finite=0, infinite=1, NaN=2), a base-2 big-endian integer
// coefficient (also known as a significand) as a []byte, and an int32 exponent.
// These are composed into a final value as "decimal = (neg) (form=finite) coefficient * 10 ^ exponent".
// A zero length coefficient is a zero value.
// The big-endian integer coefficient stores the most significant byte first (at coefficient[0]).
// If the form is not finite the coefficient and exponent should be ignored.
// The negative parameter may be set to true for any form, although implementations are not required
// to respect the negative parameter in the non-finite form.
//
// Implementations may choose to set the negative parameter to true on a zero or NaN value,
// but implementations that do not differentiate between negative and positive
// zero or NaN values should ignore the negative parameter without error.
// If an implementation does not support Infinity it may be converted into a NaN without error.
// If a value is set that is larger than what is supported by an implementation,
// an error must be returned.
// Implementations must return an error if a NaN or Infinity is attempted to be set while neither
// are supported.
//
// NOTE(kardianos): This is an experimental interface. See https://golang.org/issue/30870
type decimal interface {
	decimalDecompose
	decimalCompose
}

type decimalDecompose interface {
	// Decompose returns the internal decimal state in parts.
	// If the provided buf has sufficient capacity, buf may be returned as the coefficient with
	// the value set and length set as appropriate.
	Decompose(buf []byte) (form byte, negative bool, coefficient []byte, exponent int32)
}

type decimalCompose interface {
	// Compose sets the internal decimal value from parts. If the value cannot be
	// represented then an error should be returned.
	Compose(form byte, negative bool, coefficient []byte, exponent int32) error
}
