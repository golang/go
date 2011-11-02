// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package patch

import "os"

// An Op is a single operation to execute to apply a patch.
type Op struct {
	Verb Verb   // action
	Src  string // source file
	Dst  string // destination file
	Mode int    // mode for destination (if non-zero)
	Data []byte // data for destination (if non-nil)
}

// Apply applies the patch set to the files named in the patch set,
// constructing an in-memory copy of the new file state.
// It is the client's job to write the changes to the file system
// if desired.
//
// The function readFile should return the contents of the named file.
// Typically this function will be io.ReadFile.
//
func (set *Set) Apply(readFile func(string) ([]byte, error)) ([]Op, error) {
	op := make([]Op, len(set.File))

	for i, f := range set.File {
		o := &op[i]
		o.Verb = f.Verb
		o.Src = f.Src
		o.Dst = f.Dst
		o.Mode = f.NewMode
		if f.Diff != NoDiff || o.Verb != Edit {
			// Clients assume o.Data == nil means no data diff.
			// Start with a non-nil data.
			var old []byte = make([]byte, 0) // not nil
			var err error
			if f.Src != "" {
				old, err = readFile(f.Src)
				if err != nil {
					return nil, &os.PathError{string(f.Verb), f.Src, err}
				}
			}
			o.Data, err = f.Diff.Apply(old)
			if err != nil {
				return nil, &os.PathError{string(f.Verb), f.Src, err}
			}
		}
	}

	return op, nil
}
