package filepath

import (
	"os"
	"strings"
	"syscall"
)

// HasPrefix exists for historical compatibility and should not be used.
//
// Deprecated: HasPrefix does not respect path boundaries and
// does not ignore case when required.
func HasPrefix(p, prefix string) bool {
	if strings.HasPrefix(p, prefix) {
		return true
	}
	return strings.HasPrefix(strings.ToLower(p), strings.ToLower(prefix))
}

func splitList(path string) []string {
	// The same implementation is used in LookPath in os/exec;
	// consider changing os/exec when changing this.

	if path == "" {
		return []string{}
	}

	// Split path, respecting but preserving quotes.
	list := []string{}
	start := 0
	quo := false
	for i := 0; i < len(path); i++ {
		switch c := path[i]; {
		case c == '"':
			quo = !quo
		case c == ListSeparator && !quo:
			list = append(list, path[start:i])
			start = i + 1
		}
	}
	list = append(list, path[start:])

	// Remove quotes.
	for i, s := range list {
		list[i] = strings.ReplaceAll(s, `"`, ``)
	}

	return list
}

func abs(path string) (string, error) {
	if path == "" {
		// syscall.FullPath returns an error on empty path, because it's not a valid path.
		// To implement Abs behavior of returning working directory on empty string input,
		// special-case empty path by changing it to "." path. See golang.org/issue/24441.
		path = "."
	}
	fullPath, err := syscall.FullPath(path)
	if err != nil {
		return "", err
	}
	return Clean(fullPath), nil
}

func join(elem []string) string {
	var b strings.Builder
	var lastChar byte
	for _, e := range elem {
		switch {
		case b.Len() == 0:
			// Add the first non-empty path element unchanged.
		case os.IsPathSeparator(lastChar):
			// If the path ends in a slash, strip any leading slashes from the next
			// path element to avoid creating a UNC path (any path starting with "\\")
			// from non-UNC elements.
			//
			// The correct behavior for Join when the first element is an incomplete UNC
			// path (for example, "\\") is underspecified. We currently join subsequent
			// elements so Join("\\", "host", "share") produces "\\host\share".
			for len(e) > 0 && os.IsPathSeparator(e[0]) {
				e = e[1:]
			}
			// If the path is \ and the next path element is ??,
			// add an extra .\ to create \.\?? rather than \??\
			// (a Root Local Device path).
			if b.Len() == 1 && strings.HasPrefix(e, "??") && (len(e) == len("??") || os.IsPathSeparator(e[2])) {
				b.WriteString(`.\`)
			}
		case lastChar == ':':
			// If the path ends in a colon, keep the path relative to the current directory
			// on a drive and don't add a separator. Preserve leading slashes in the next
			// path element, which may make the path absolute.
			//
			// 	Join(`C:`, `f`) = `C:f`
			//	Join(`C:`, `\f`) = `C:\f`
		default:
			// In all other cases, add a separator between elements.
			b.WriteByte('\\')
			lastChar = '\\'
		}
		if len(e) > 0 {
			b.WriteString(e)
			lastChar = e[len(e)-1]
		}
	}
	if b.Len() == 0 {
		return ""
	}
	return Clean(b.String())
}

func sameWord(a, b string) bool {
	return strings.EqualFold(a, b)
}
