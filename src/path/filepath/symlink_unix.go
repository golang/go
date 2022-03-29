//go:build !windows

package filepath

func evalSymlinks(path string) (string, error) {
	return walkSymlinks(path)
}
