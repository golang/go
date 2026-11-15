// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

/*
Package simd implements portable and vector-size-agnostic SIMD types,
and functions and methods for working with these types.  SIMD types
are either implemented in hardware (for example, arm64 "Neon" or
amd64 "AVX/AVX2/AVX512") using the corresponding types in the
[simd/archsimd] package, or emulated in pure Go.  In all cases, the
vector length is at least 128 bits, and within a given program
execution, all vectors have the same length.

# SIMD Types

There is a simd type corresponding to each primitive numeric type,
except for complex64 and complex128.  Each of these is the type name,
capitalized, with an "s" suffix, for example [Int8s], [Uint16s],
or [Float64s].

There are also simd "mask" types that abstract the mask registers
present in some architectures, otherwise these will be implemented
as bit masks.

# Obtaining SIMD values

The zero value of a SIMD vector type is valid and represents a zero vector.

For each SIMD type, "Load<Types>(s []<type>) <Types>"
loads a full vector of the type from a long-enough slice.

For slices that are not long enough, "Load<Types>Part(s []<type>) (<Types>, int)"
will load as many elements as are available from the slice, fill the remainder
with zero, and also return the number that were loaded.

For each SIMD type, "Broadcast<Types>(x type) <Types>"
returns a vector whose elements are all initialized to x.

Examples:
  - [LoadInt8s]
  - [LoadUint16sPart]
  - [BroadcastFloat32s]

# Operations

SIMD types provide methods for unary ([Float32s.Abs], [Int16s.Not]), binary
([Float64s.Add], [Uint32s.GreaterEqual]), and ternary operations ([Int64s.IfElse], [Float32s.MulAdd]).
Relational operations produce masks.

SIMD types also support conversions between types, both those that are
mostly value-preserving ([Int32s.ConvertToFloat32], [Float32s.ConvertToInt32]) and
those that change types without altering the underlying vector bit
pattern.

Signed integer types convert to mask types ([Int16s.ToMask]), but this is a
comparison against zero, not a simple bitwise conversion.  Mask types
convert to signed integers in an operation ([Mask32s.ToInt32s]) that may be a
zero-cost bitwise conversion, or not, depending on the underlying hardware.

# Equality

Vector and mask types are comparable with == and !=. For comparison, a vector
behaves like a struct with one field per element, and a mask like a struct
with one bool per element. Like regular float comparison, NaNs are never equal
and +0 equals -0.

# Storing

SIMD vector types have two methods, one for storing the entire vector into a slice
"Store([]<type>)" and a second for storing part of a vector into a slice "StorePart([]<type>) int".
StorePart returns the number of elements actually stored.

# String conversion

Vectors and masks provide a String method for conversion to strings.

# Conversion to and from simd/archsimd types

Each SIMD vector and mask type has a ToArch() any method that returns the
[simd/archsimd] value backing the receiver, or nil if [Emulated] is true.
Callers inspecting the result for architecture-specific types must be
build-tagged to the relevant architecture(s) and use a type switch with a
default case.

On architectures supported by [simd/archsimd], the simd package also provides
generic FromArch functions (such as Float32sFromArch) that convert a
[simd/archsimd] vector or mask (e.g. [archsimd.Float32x4]) into the
corresponding simd type. They panic if [Emulated] is true or if the input is
not for vector width [VectorBits].

See the [Int16s.ToArch] example for converting between [simd] and
[simd/archsimd] types.
*/
package simd

// BUG(reflection): Calls won't work, and there may be other bugs.
// BUG(global initialization): SIMD-dependent var initializers don't work.
// BUG(modified names): Modified names may appear in stack traces and debugging.
