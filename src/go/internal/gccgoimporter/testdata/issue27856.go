package lib

type M struct {
	E E
}
type F struct {
	_ *M
}
type E = F
