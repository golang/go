package fsys

import (
	"io/fs"
	"path/filepath"
	"strings"
	"sync"
)

// SourceExpander is installed by the GoX command adapter. The filesystem package
// has no dependency on the language front end or command configuration.
var SourceExpander func(string, []fs.DirEntry) ([]fs.DirEntry, error)
var generatedSources sync.Map

func RegisterGenerated(virtual, actual string) { generatedSources.Store(abs(virtual), actual) }
func Generated(name string) (string, bool) {
	v, ok := generatedSources.Load(abs(name))
	if !ok {
		return "", false
	}
	return v.(string), true
}
func HasGoX(dir string) bool {
	entries, e := readDir(dir)
	if e != nil {
		return false
	}
	for _, d := range entries {
		if !d.IsDir() && strings.HasSuffix(d.Name(), ".gox") {
			return true
		}
	}
	return false
}
func GoXOriginal(name string) string {
	if _, ok := Generated(name); ok {
		return strings.TrimSuffix(name, filepath.Ext(name)) + ".gox"
	}
	return ""
}
