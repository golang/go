// errorcheckdir

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A constraint interface that a plain Go package declares but never
// names keeps its type terms in the export data, so an importer still
// rejects type arguments outside the type set. Exporting the interface
// without its terms would make it the empty interface and accept every
// type.

package ignored
