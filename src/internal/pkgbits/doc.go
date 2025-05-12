// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
The Unified IR (UIR) format for primitive types is implicitly defined by the
package pkgbits.

The most basic primitives are laid out as below.

Bool    = [ Sync ] byte .
Int64   = [ Sync ] zvarint .
Uint64  = [ Sync ] uvarint .

zvarint = (* a zig-zag encoded signed variable-width integer *) .
uvarint = (* an unsigned variable-width integer *) .

# Strings
A string is a series of bytes.

// TODO(markfreeman): Does this need a marker?
String    = { byte } .

Strings are typically not encoded directly. Rather, they are deduplicated
during encoding and referenced where needed; this process is called interning.

StringRef = [ Sync ] Ref[String] .

Note that StringRef is *not* equivalent to Ref[String] due to the extra marker.

# References
References specify the location of a value. While the representation here is
fixed, the interpretation of a reference is left to other packages.

Ref[T] = [ Sync ] Uint64 . // points to a value of type T

# Slices
Slices are a convenience for encoding a series of values of the same type.

// TODO(markfreeman): Does this need a marker?
Slice[T] = Uint64 // the number of values in the slice
           { T }  // the values
           .

# Constants
Constants appear as defined via the package constant.

Constant = [ Sync ]
           Bool        // whether the constant is a complex number
           Scalar      // the real part
           [ Scalar ]  // if complex, the imaginary part
           .

A scalar represents a value using one of several potential formats. The exact
format and interpretation is distinguished by a code preceding the value.

Scalar   = [ Sync ]
           Uint64      // the code indicating the type of Val
           Val
           .

Val      = Bool
         | Int64
         | StringRef
         | Term        // big integer
         | Term Term   // big ratio, numerator / denominator
         | BigBytes    // big float, precision 512
           .

Term     = BigBytes
           Bool        // whether the term is negative
           .

BigBytes = StringRef . // bytes of a big value

# Markers
Markers provide a mechanism for asserting that encoders and decoders are
synchronized. If an unexpected marker is found, decoding panics.

Sync = uvarint          // indicates what should follow if synchronized
       WriterPCs
       .

A marker also records a configurable number of program counters (PCs) during
encoding to assist with debugging.

WriterPCs = uvarint     // the number of PCs that follow
            { uvarint } // the PCs
            .

Note that markers are always defined using terminals — they never contain a
marker themselves.
*/

// Package pkgbits implements low-level coding abstractions for Unified IR's
// (UIR) binary export data format.
//
// At a low-level, the exported objects of a package are encoded as a byte
// array. This array contains byte representations of primitive, potentially
// variable-length values, such as integers, booleans, strings, and constants.
//
// Additionally, the array may contain values which denote indices in the byte
// array itself. These are termed "relocations" and allow for references.
//
// The details of mapping high-level Go constructs to primitives are left to
// other packages.
package pkgbits
