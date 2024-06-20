//go:build faketime && linux && arm64

package runtime

func getActualTime() int64 {
	sec, nsec := walltime()
	t := int64(secs)*1e9 + int64(nsecs)
	return t
}
