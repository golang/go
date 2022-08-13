//go:build !windows && !plan9

package filepath

func evalSymlinks(path string) (string, error) {
	return walkSymlinks(path)
}
