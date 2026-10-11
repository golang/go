package gox

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"sync"
)

// Include the actual executable in cache identity so development builds and
// vendored engines never reuse output produced by an older transformer.
var executableFingerprint = sync.OnceValues(func() (string, error) {
	path, e := os.Executable()
	if e != nil {
		return "", e
	}
	f, e := os.Open(path)
	if e != nil {
		return "", e
	}
	defer f.Close()
	h := sha256.New()
	if _, e = io.Copy(h, f); e != nil {
		return "", e
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
})
