//go:build faketime && linux && amd64

package runtime

func actualTime() (sec int64, nsec int32, mono int64)

func getActualTime() int64 {
	secs, nsecs, _ := actualTime()
	t := int64(secs)*1e9 + int64(nsecs)
	return t
}
