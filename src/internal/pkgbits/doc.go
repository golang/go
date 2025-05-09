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
Strings are not encoded directly. Rather, they are deduplicated during encoding
and referenced where needed.

String      = [ Sync ] StringRef .
StringRef   = [ Sync ] Uint64 . // TODO(markfreeman): Document.

StringSlice = Uint64            // the number of strings in the slice
              { String }
              .

// TODO(markfreeman) It is awkward to discuss references (and by extension
// strings and constants). We cannot explain how they resolve without mention
// of foreign concepts. Ideally, references would be defined in familar terms —
// perhaps using an index on the byte array.

# Constants
Constants appear as defined via the package constant.

Constant = [ Sync ]
           Bool       // whether the constant is a complex number
           Scalar     // the real part
           [ Scalar ] // if complex, the imaginary part
           .

A scalar represents a value using one of several potential formats. The exact
format and interpretation is distinguished by a code preceding the value.

Scalar   = [ Sync ]
           Uint64     // the code
           Val
           .

Val      = Bool
         | Int64
         | String
         | Term       // big integer
         | Term Term  // big ratio, numerator / denominator
         | BigBytes   // big float, precision 512
           .

Term     = BigBytes
           Bool       // whether the term is negative
           .

BigBytes = String .   // bytes of a big value

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
