// skip

package outsider

func Launch(fn func()) {
	go fn()
}
