package modindex

import (
	"io/fs"
	"path/filepath"
	"strings"
	"sync"
)

var goxModules sync.Map

// Module indexes created by older tools do not record .gox source. Module-cache
// trees are immutable, so one raw directory scan per module is sufficient.
func moduleHasGoX(root string) bool {
	if v, ok := goxModules.Load(root); ok {
		return v.(bool)
	}
	found := false
	_ = filepath.WalkDir(root, func(path string, d fs.DirEntry, e error) error {
		if e != nil {
			return nil
		}
		if !d.IsDir() && strings.HasSuffix(d.Name(), ".gox") {
			found = true
			return fs.SkipAll
		}
		return nil
	})
	goxModules.Store(root, found)
	return found
}
