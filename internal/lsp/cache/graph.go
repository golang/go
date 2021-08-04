// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import "golang.org/x/tools/internal/span"

// A metadataGraph holds information about a transtively closed import graph of
// Go packages, as obtained from go/packages.
//
// Currently a new metadata graph is created for each snapshot.
// TODO(rfindley): make this type immutable, so that it may be shared across
// snapshots.
type metadataGraph struct {
	// ids maps file URIs to package IDs. A single file may belong to multiple
	// packages due to tests packages.
	ids map[span.URI][]PackageID

	// metadata maps package IDs to their associated metadata.
	metadata map[PackageID]*KnownMetadata

	// importedBy maps package IDs to the list of packages that import them.
	importedBy map[PackageID][]PackageID
}

func NewMetadataGraph() *metadataGraph {
	return &metadataGraph{
		ids:        make(map[span.URI][]PackageID),
		metadata:   make(map[PackageID]*KnownMetadata),
		importedBy: make(map[PackageID][]PackageID),
	}
}
