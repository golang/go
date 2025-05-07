// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
The Unified IR (UIR) format is implicitly defined by the package noder.

At the highest level, a package encoded in UIR follows the grammar below.

File        = Header Payload fingerprint .
Header      = version [ flags ] sectionEnds elementEnds .

version     = uint32 .     // used for backward compatibility
flags       = uint32 .     // feature flags used across versions
sectionEnds = [10]uint32 . // defines section boundaries
elementEnds = []uint32 .   // defines element boundaries
fingerprint = [8]byte .    // sha256 fingerprint

The payload has a structure as well. It is a series of sections, which
contain elements of the same type. Go constructs are mapped onto
(potentially multiple) elements. It is represented as below.

TODO(markfreeman): Update when we rename RelocFoo to SectionFoo.
Payload = RelocString
          RelocMeta
          RelocPosBase
          RelocPkg
          RelocName
          RelocType
          RelocObj
          RelocObjExt
          RelocObjDict
          RelocBody
          .
*/

package noder
