// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package quick implements utility functions to help with black box testing.
package quick

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"strings"
)

var defaultMaxCount *int = flag.Int("quickchecks", 100, "The default number of iterations for each check")

// A Generator can generate random values of its own type.
type Generator interface {
	// Generate returns a random instance of the type on which it is a
	// method using the size as a size hint.
	Generate(rand *rand.Rand, size int) reflect.Value
}

// randFloat32 generates a random float taking the full range of a float32.
func randFloat32(rand *rand.Rand) float32 {
	f := rand.Float64() * math.MaxFloat32
	if rand.Int()&1 == 1 {
		f = -f
	}
	return float32(f)
}

// randFloat64 generates a random float taking the full range of a float64.
func randFloat64(rand *rand.Rand) float64 {
	f := rand.Float64()
	if rand.Int()&1 == 1 {
		f = -f
	}
	return f
}

// randInt64 returns a random integer taking half the range of an int64.
func randInt64(rand *rand.Rand) int64 { return rand.Int63() - 1<<62 }

// complexSize is the maximum length of arbitrary values that contain other
// values.
const complexSize = 50

// Value returns an arbitrary value of the given type.
// If the type implements the Generator interface, that will be used.
// Note: in order to create arbitrary values for structs, all the members must be public.
func Value(t reflect.Type, rand *rand.Rand) (value reflect.Value, ok bool) {
	if m, ok := reflect.Zero(t).Interface().(Generator); ok {
		return m.Generate(rand, complexSize), true
	}

	switch concrete := t; concrete.Kind() {
	case reflect.Bool:
		return reflect.ValueOf(rand.Int()&1 == 0), true
	case reflect.Float32:
		return reflect.ValueOf(randFloat32(rand)), true
	case reflect.Float64:
		return reflect.ValueOf(randFloat64(rand)), true
	case reflect.Complex64:
		return reflect.ValueOf(complex(randFloat32(rand), randFloat32(rand))), true
	case reflect.Complex128:
		return reflect.ValueOf(complex(randFloat64(rand), randFloat64(rand))), true
	case reflect.Int16:
		return reflect.ValueOf(int16(randInt64(rand))), true
	case reflect.Int32:
		return reflect.ValueOf(int32(randInt64(rand))), true
	case reflect.Int64:
		return reflect.ValueOf(randInt64(rand)), true
	case reflect.Int8:
		return reflect.ValueOf(int8(randInt64(rand))), true
	case reflect.Int:
		return reflect.ValueOf(int(randInt64(rand))), true
	case reflect.Uint16:
		return reflect.ValueOf(uint16(randInt64(rand))), true
	case reflect.Uint32:
		return reflect.ValueOf(uint32(randInt64(rand))), true
	case reflect.Uint64:
		return reflect.ValueOf(uint64(randInt64(rand))), true
	case reflect.Uint8:
		return reflect.ValueOf(uint8(randInt64(rand))), true
	case reflect.Uint:
		return reflect.ValueOf(uint(randInt64(rand))), true
	case reflect.Uintptr:
		return reflect.ValueOf(uintptr(randInt64(rand))), true
	case reflect.Map:
		numElems := rand.Intn(complexSize)
		m := reflect.MakeMap(concrete)
		for i := 0; i < numElems; i++ {
			key, ok1 := Value(concrete.Key(), rand)
			value, ok2 := Value(concrete.Elem(), rand)
			if !ok1 || !ok2 {
				return reflect.Value{}, false
			}
			m.SetMapIndex(key, value)
		}
		return m, true
	case reflect.Ptr:
		v, ok := Value(concrete.Elem(), rand)
		if !ok {
			return reflect.Value{}, false
		}
		p := reflect.New(concrete.Elem())
		p.Elem().Set(v)
		return p, true
	case reflect.Slice:
		numElems := rand.Intn(complexSize)
		s := reflect.MakeSlice(concrete, numElems, numElems)
		for i := 0; i < numElems; i++ {
			v, ok := Value(concrete.Elem(), rand)
			if !ok {
				return reflect.Value{}, false
			}
			s.Index(i).Set(v)
		}
		return s, true
	case reflect.String:
		numChars := rand.Intn(complexSize)
		codePoints := make([]rune, numChars)
		for i := 0; i < numChars; i++ {
			codePoints[i] = rune(rand.Intn(0x10ffff))
		}
		return reflect.ValueOf(string(codePoints)), true
	case reflect.Struct:
		s := reflect.New(t).Elem()
		for i := 0; i < s.NumField(); i++ {
			v, ok := Value(concrete.Field(i).Type, rand)
			if !ok {
				return reflect.Value{}, false
			}
			s.Field(i).Set(v)
		}
		return s, true
	default:
		return reflect.Value{}, false
	}

	return
}

// A Config structure contains options for running a test.
type Config struct {
	// MaxCount sets the maximum number of iterations. If zero,
	// MaxCountScale is used.
	MaxCount int
	// MaxCountScale is a non-negative scale factor applied to the default
	// maximum. If zero, the default is unchanged.
	MaxCountScale float64
	// If non-nil, rand is a source of random numbers. Otherwise a default
	// pseudo-random source will be used.
	Rand *rand.Rand
	// If non-nil, Values is a function which generates a slice of arbitrary
	// Values that are congruent with the arguments to the function being
	// tested. Otherwise, Values is used to generate the values.
	Values func([]reflect.Value, *rand.Rand)
}

var defaultConfig Config

// getRand returns the *rand.Rand to use for a given Config.
func (c *Config) getRand() *rand.Rand {
	if c.Rand == nil {
		return rand.New(rand.NewSource(0))
	}
	return c.Rand
}

// getMaxCount returns the maximum number of iterations to run for a given
// Config.
func (c *Config) getMaxCount() (maxCount int) {
	maxCount = c.MaxCount
	if maxCount == 0 {
		if c.MaxCountScale != 0 {
			maxCount = int(c.MaxCountScale * float64(*defaultMaxCount))
		} else {
			maxCount = *defaultMaxCount
		}
	}

	return
}

// A SetupError is the result of an error in the way that check is being
// used, independent of the functions being tested.
type SetupError string

func (s SetupError) Error() string { return string(s) }

// A CheckError is the result of Check finding an error.
type CheckError struct {
	Count int
	In    []interface{}
}

func (s *CheckError) Error() string {
	return fmt.Sprintf("#%d: failed on input %s", s.Count, toString(s.In))
}

// A CheckEqualError is the result CheckEqual finding an error.
type CheckEqualError struct {
	CheckError
	Out1 []interface{}
	Out2 []interface{}
}

func (s *CheckEqualError) Error() string {
	return fmt.Sprintf("#%d: failed on input %s. Output 1: %s. Output 2: %s", s.Count, toString(s.In), toString(s.Out1), toString(s.Out2))
}

// Check looks for an input to f, any function that returns bool,
// such that f returns false.  It calls f repeatedly, with arbitrary
// values for each argument.  If f returns false on a given input,
// Check returns that input as a *CheckError.
// For example:
//
// 	func TestOddMultipleOfThree(t *testing.T) {
// 		f := func(x int) bool {
// 			y := OddMultipleOfThree(x)
// 			return y%2 == 1 && y%3 == 0
// 		}
// 		if err := quick.Check(f, nil); err != nil {
// 			t.Error(err)
// 		}
// 	}
func Check(function interface{}, config *Config) (err error) {
	if config == nil {
		config = &defaultConfig
	}

	f, fType, ok := functionAndType(function)
	if !ok {
		err = SetupError("argument is not a function")
		return
	}

	if fType.NumOut() != 1 {
		err = SetupError("function returns more than one value.")
		return
	}
	if fType.Out(0).Kind() != reflect.Bool {
		err = SetupError("function does not return a bool")
		return
	}

	arguments := make([]reflect.Value, fType.NumIn())
	rand := config.getRand()
	maxCount := config.getMaxCount()

	for i := 0; i < maxCount; i++ {
		err = arbitraryValues(arguments, fType, config, rand)
		if err != nil {
			return
		}

		if !f.Call(arguments)[0].Bool() {
			err = &CheckError{i + 1, toInterfaces(arguments)}
			return
		}
	}

	return
}

// CheckEqual looks for an input on which f and g return different results.
// It calls f and g repeatedly with arbitrary values for each argument.
// If f and g return different answers, CheckEqual returns a *CheckEqualError
// describing the input and the outputs.
func CheckEqual(f, g interface{}, config *Config) (err error) {
	if config == nil {
		config = &defaultConfig
	}

	x, xType, ok := functionAndType(f)
	if !ok {
		err = SetupError("f is not a function")
		return
	}
	y, yType, ok := functionAndType(g)
	if !ok {
		err = SetupError("g is not a function")
		return
	}

	if xType != yType {
		err = SetupError("functions have different types")
		return
	}

	arguments := make([]reflect.Value, xType.NumIn())
	rand := config.getRand()
	maxCount := config.getMaxCount()

	for i := 0; i < maxCount; i++ {
		err = arbitraryValues(arguments, xType, config, rand)
		if err != nil {
			return
		}

		xOut := toInterfaces(x.Call(arguments))
		yOut := toInterfaces(y.Call(arguments))

		if !reflect.DeepEqual(xOut, yOut) {
			err = &CheckEqualError{CheckError{i + 1, toInterfaces(arguments)}, xOut, yOut}
			return
		}
	}

	return
}

// arbitraryValues writes Values to args such that args contains Values
// suitable for calling f.
func arbitraryValues(args []reflect.Value, f reflect.Type, config *Config, rand *rand.Rand) (err error) {
	if config.Values != nil {
		config.Values(args, rand)
		return
	}

	for j := 0; j < len(args); j++ {
		var ok bool
		args[j], ok = Value(f.In(j), rand)
		if !ok {
			err = SetupError(fmt.Sprintf("cannot create arbitrary value of type %s for argument %d", f.In(j), j))
			return
		}
	}

	return
}

func functionAndType(f interface{}) (v reflect.Value, t reflect.Type, ok bool) {
	v = reflect.ValueOf(f)
	ok = v.Kind() == reflect.Func
	if !ok {
		return
	}
	t = v.Type()
	return
}

func toInterfaces(values []reflect.Value) []interface{} {
	ret := make([]interface{}, len(values))
	for i, v := range values {
		ret[i] = v.Interface()
	}
	return ret
}

func toString(interfaces []interface{}) string {
	s := make([]string, len(interfaces))
	for i, v := range interfaces {
		s[i] = fmt.Sprintf("%#v", v)
	}
	return strings.Join(s, ", ")
}
