package t

import "d"

func t(i int) int {
	data := d.Data{V: i}
	return d.D(i) + data.Do()
}
