package p1

type SockaddrUnix int

func (s SockaddrUnix) Error() string { return "blah" }
