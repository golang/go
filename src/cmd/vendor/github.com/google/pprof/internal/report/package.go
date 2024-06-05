package report

import "regexp"

// pkgRE extracts package name, It looks for the first "." or "::" that occurs
// after the last "/". (Searching after the last / allows us to correctly handle
// names that look like "some.url.com/foo.bar".)
var pkgRE = regexp.MustCompile(`^((.*/)?[\w\d_]+)(\.|::)([^/]*)$`)

// packageName returns the package name of the named symbol, or "" if not found.
func packageName(name string) string {
	m := pkgRE.FindStringSubmatch(name)
	if m == nil {
		return ""
	}
	return m[1]
}
