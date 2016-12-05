// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package quick implements utility functions to help with black box testing.
//
// The testing/quick package is frozen and is not accepting new features.
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
	f := rand.Float64() * math.MaxFloat64
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
// Note: To create arbitrary values for structs, all the fields must be exported.
func Value(t reflect.Type, rand *rand.Rand) (value reflect.Value, ok bool) {
	return sizedValue(t, rand, complexSize)
}

// sizedValue returns an arbitrary value of the given type. The size
// hint is used for shrinking as a function of indirection level so
// that recursive data structures will terminate.
func sizedValue(t reflect.Type, rand *rand.Rand, size int) (value reflect.Value, ok bool) {
	if m, ok := reflect.Zero(t).Interface().(Generator); ok {
		return m.Generate(rand, size), true
	}

	v := reflect.New(t).Elem()
	switch concrete := t; concrete.Kind() {
	case reflect.Bool:
		v.SetBool(rand.Int()&1 == 0)
	case reflect.Float32:
		v.SetFloat(float64(randFloat32(rand)))
	case reflect.Float64:
		v.SetFloat(randFloat64(rand))
	case reflect.Complex64:
		v.SetComplex(complex(float64(randFloat32(rand)), float64(randFloat32(rand))))
	case reflect.Complex128:
		v.SetComplex(complex(randFloat64(rand), randFloat64(rand)))
	case reflect.Int16:
		v.SetInt(randInt64(rand))
	case reflect.Int32:
		v.SetInt(randInt64(rand))
	case reflect.Int64:
		v.SetInt(randInt64(rand))
	case reflect.Int8:
		v.SetInt(randInt64(rand))
	case reflect.Int:
		v.SetInt(randInt64(rand))
	case reflect.Uint16:
		v.SetUint(uint64(randInt64(rand)))
	case reflect.Uint32:
		v.SetUint(uint64(randInt64(rand)))
	case reflect.Uint64:
		v.SetUint(uint64(randInt64(rand)))
	case reflect.Uint8:
		v.SetUint(uint64(randInt64(rand)))
	case reflect.Uint:
		v.SetUint(uint64(randInt64(rand)))
	case reflect.Uintptr:
		v.SetUint(uint64(randInt64(rand)))
	case reflect.Map:
		numElems := rand.Intn(size)
		v.Set(reflect.MakeMap(concrete))
		for i := 0; i < numElems; i++ {
			key, ok1 := sizedValue(concrete.Key(), rand, size)
			value, ok2 := sizedValue(concrete.Elem(), rand, size)
			if !ok1 || !ok2 {
				return reflect.Value{}, false
			}
			v.SetMapIndex(key, value)
		}
	case reflect.Ptr:
		if rand.Intn(size) == 0 {
			v.Set(reflect.Zero(concrete)) // Generate nil pointer.
		} else {
			elem, ok := sizedValue(concrete.Elem(), rand, size)
			if !ok {
				return reflect.Value{}, false
			}
			v.Set(reflect.New(concrete.Elem()))
			v.Elem().Set(elem)
		}
	case reflect.Slice:
		numElems := rand.Intn(size)
		sizeLeft := size - numElems
		v.Set(reflect.MakeSlice(concrete, numElems, numElems))
		for i := 0; i < numElems; i++ {
			elem, ok := sizedValue(concrete.Elem(), rand, sizeLeft)
			if !ok {
				return reflect.Value{}, false
			}
			v.Index(i).Set(elem)
		}
	case reflect.Array:
		for i := 0; i < v.Len(); i++ {
			elem, ok := sizedValue(concrete.Elem(), rand, size)
			if !ok {
				return reflect.Value{}, false
			}
			v.Index(i).Set(elem)
		}
	case reflect.String:
		numChars := rand.Intn(complexSize)
		codePoints := make([]rune, numChars)
		for i := 0; i < numChars; i++ {
			codePoints[i] = rune(rand.Intn(0x10ffff))
		}
		v.SetString(string(codePoints))
	case reflect.Struct:
		n := v.NumField()
		// Divide sizeLeft evenly among the struct fields.
		sizeLeft := size
		if n > sizeLeft {
			sizeLeft = 1
		} else if n > 0 {
			sizeLeft /= n
		}
		for i := 0; i < n; i++ {
			elem, ok := sizedValue(concrete.Field(i).Type, rand, sizeLeft)
			if !ok {
				return reflect.Value{}, false
			}
			v.Field(i).Set(elem)
		}
	default:
		return reflect.Value{}, false
	}

	return v, true
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
	// If non-nil, the Values function generates a slice of arbitrary
	// reflect.Values that are congruent with the arguments to the function
	// being tested. Otherwise, the top-level Value function is used
	// to generate them.
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
// such that f returns false. It calls f repeatedly, with arbitrary
// values for each argument. If f returns false on a given input,
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
func Check(f interface{}, config *Config) error {
	if config == nil {
		config = &defaultConfig
	}

	fVal, fType, ok := functionAndType(f)
	if !ok {
		return SetupError("argument is not a function")
	}

	if fType.NumOut() != 1 {
		return SetupError("function does not return one value")
	}
	if fType.Out(0).Kind() != reflect.Bool {
		return SetupError("function does not return a bool")
	}

	arguments := make([]reflect.Value, fType.NumIn())
	rand := config.getRand()
	maxCount := config.getMaxCount()

	for i := 0; i < maxCount; i++ {
		err := arbitraryValues(arguments, fType, config, rand)
		if err != nil {
			return err
		}

		if !fVal.Call(arguments)[0].Bool() {
			return &CheckError{i + 1, toInterfaces(arguments)}
		}
	}

	return nil
}

// CheckEqual looks for an input on which f and g return different results.
// It calls f and g repeatedly with arbitrary values for each argument.
// If f and g return different answers, CheckEqual returns a *CheckEqualError
// describing the input and the outputs.
func CheckEqual(f, g interface{}, config *Config) error {
	if config == nil {
		config = &defaultConfig
	}

	x, xType, ok := functionAndType(f)
	if !ok {
		return SetupError("f is not a function")
	}
	y, yType, ok := functionAndType(g)
	if !ok {
		return SetupError("g is not a function")
	}

	if xType != yType {
		return SetupError("functions have different types")
	}

	arguments := make([]reflect.Value, xType.NumIn())
	rand := config.getRand()
	maxCount := config.getMaxCount()

	for i := 0; i < maxCount; i++ {
		err := arbitraryValues(arguments, xType, config, rand)
		if err != nil {
			return err
		}

		xOut := toInterfaces(x.Call(arguments))
		yOut := toInterfaces(y.Call(arguments))

		if !reflect.DeepEqual(xOut, yOut) {
			return &CheckEqualError{CheckError{i + 1, toInterfaces(arguments)}, xOut, yOut}
		}
	}

	return nil
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
